import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import tkinter.font as tkfont
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import tensorflow as tf
from threading import Thread
from queue import Queue
import concurrent.futures
import threading
from functools import wraps
from datetime import datetime

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def worker():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = Thread(target=worker)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
            if error[0] is not None:
                raise error[0]
            return result[0]
        return wrapper
    return decorator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='predictor.log'
)

# ----------------- Utilities -----------------
def parse_draw(combo_str):
    return [int(n) for n in combo_str.split("-")]

def build_history(df):
    """Return list of sets: draws[t] is a set of numbers in draw t."""
    try:
        draws = []
        for combo in df["Combinations"]:
            draws.append(set(parse_draw(combo)))
        return draws
    except Exception as e:
        logging.error(f"Error in build_history: {str(e)}")
        return []

# ----------------- Feature Engineering -----------------
def process_number_features(k, draws, lookback_short, lookback_long, decay):
    """Process features for a single number"""
    rows = []
    targets = []
    T = len(draws)

    # Running decayed counts per number
    decayed = 0.0

    # For rolling windows, keep FIFO buffers of counts per number
    from collections import deque
    win_short = deque(maxlen=lookback_short)
    win_long  = deque(maxlen=lookback_long)

    for t in range(T):
        current = draws[t]
        # Update structures with current draw for next step's features
        appeared = 1 if k in current else 0
        decayed = decay * decayed + appeared
        win_short.append(appeared)
        win_long.append(appeared)

        # We can only form a labeled example for time t (the draw that already happened)
        # using features computed from history BEFORE t. So features for predicting t use draws[:t].
        # To predict t+1 later, we will extract with history up to T-1.
        if t == 0:
            continue  # need at least 1 step of history

        # Build rows for time t using history up to t-1
        # Rolling sums (short, long)
        short_sum = sum(win_short) if len(win_short) > 0 else 0.0
        long_sum  = sum(win_long)  if len(win_long)  > 0 else 0.0
        short_rate = short_sum / max(1, len(win_short))
        long_rate  = long_sum  / max(1, len(win_long))

        # Time since last seen (capped for stability)
        tsl = (t - 1) if t - 1 != -1 else lookback_long + 5
        tsl = min(tsl, lookback_long + 5)

        # Momentum: short vs long difference
        momentum = short_rate - long_rate

        rows.append([k, short_sum, long_sum, short_rate, long_rate,
                     decayed, tsl, momentum])
        targets.append(1 if k in draws[t] else 0)

    return rows, targets

def per_number_features(draws, lookback_short=10, lookback_long=50, decay=0.96):
    """
    Build a panel dataset:
      rows: (t, k) for t in [1..T-1], k in [1..58]
      y: 1 if k drawn at time t, else 0
      X: per-number features computed from draws[:t]
    """
    try:
        T = len(draws)
        K = 58
        all_rows = []
        all_targets = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for k in range(1, K + 1):
                future = executor.submit(
                    process_number_features, 
                    k, draws, lookback_short, lookback_long, decay
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                rows, targets = future.result()
                if rows and targets:
                    all_rows.extend(rows)
                    all_targets.extend(targets)
        
        return np.array(all_rows, dtype=float), np.array(all_targets, dtype=float)
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        return None, None

# ----------------- Model -----------------
def build_model(input_dim):
    if input_dim <= 0:
        raise ValueError("Input dimension must be positive")
    
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def temperature_scale(p, T):
    # p in (0,1), T > 0; scaling in logit space
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    scaled = 1 / (1 + np.exp(-logit / max(T, eps)))
    return scaled

def fit_temperature(p_val, y_val):
    # simple grid search for T that minimizes log loss
    eps = 1e-7
    p_val = np.clip(p_val, eps, 1 - eps)
    Ts = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best_T, best_loss = 1.0, 1e9
    for T in Ts:
        q = temperature_scale(p_val, T)
        loss = -(y_val * np.log(q + eps) + (1 - y_val) * np.log(1 - q + eps)).mean()
        if loss < best_loss:
            best_loss, best_T = loss, T
    return best_T

# ----------------- Inference features for next draw -----------------
def next_draw_features(draws, lookback_short=10, lookback_long=50, decay=0.96):
    """
    Build the same feature set for time T (next draw) for all k=1..58
    using history up to T-1.
    """
    from collections import deque
    T = len(draws)
    K = 58
    last_seen = [-1] * (K + 1)
    decayed = np.zeros(K + 1, dtype=float)
    win_short = [deque(maxlen=lookback_short) for _ in range(K + 1)]
    win_long  = [deque(maxlen=lookback_long)  for _ in range(K + 1)]

    for t in range(T):
        current = draws[t]
        for k in range(1, K + 1):
            appeared = 1 if k in current else 0
            decayed[k] = decay * decayed[k] + appeared
            win_short[k].append(appeared)
            win_long[k].append(appeared)
            if appeared:
                last_seen[k] = t

    X_next = []
    for k in range(1, K + 1):
        short_sum = sum(win_short[k]) if len(win_short[k]) > 0 else 0.0
        long_sum  = sum(win_long[k])  if len(win_long[k])  > 0 else 0.0
        short_rate = short_sum / max(1, len(win_short[k]))
        long_rate  = long_sum  / max(1, len(win_long[k]))
        decayed_freq = decayed[k]
        tsl = (T - 1 - last_seen[k]) if last_seen[k] != -1 else lookback_long + 5
        tsl = min(tsl, lookback_long + 5)
        momentum = short_rate - long_rate
        X_next.append([k, short_sum, long_sum, short_rate, long_rate,
                       decayed_freq, tsl, momentum])
    return np.array(X_next, dtype=float)

# ----------------- Sampling -----------------
def sample_six(prob, enforce_constraints=True, max_tries=1000):
    """
    Sample 6 numbers without replacement from probabilities prob[1..58].
    Optionally enforce soft realism constraints: even count in [2,4], min spread >= 10.
    """
    prob = np.array(prob, dtype=float)
    prob = prob / prob.sum()
    nums = np.arange(1, 59)

    def draw_once():
        chosen = []
        p = prob.copy()
        available = nums.copy()
        for _ in range(6):
            p = p / p.sum()
            pick = np.random.choice(available, p=p)
            chosen.append(int(pick))
            # remove chosen
            idx = np.where(available == pick)[0][0]
            available = np.delete(available, idx)
            p = np.delete(p, idx)
        chosen.sort()
        return chosen

    if not enforce_constraints:
        return sorted(np.random.choice(nums, size=6, replace=False, p=prob).tolist())

    for _ in range(max_tries):
        c = draw_once()
        even = sum(1 for x in c if x % 2 == 0)
        spread = max(c) - min(c)
        if 2 <= even <= 4 and spread >= 10:
            return c
    # fallback
    return draw_once()

# ----------------- End-to-end generation -----------------
@timeout(30)  # 30 second timeout
def generate_combination():
    try:
        df = pd.read_csv("combinations.csv")
        if df.empty:
            raise ValueError("combinations.csv is empty")
    except pd.errors.EmptyDataError:
        return "Error: combinations.csv is empty", 0.0
    except FileNotFoundError:
        return "Error: combinations.csv not found. Run csvcreation.py first.", 0.0
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", 0.0

    try:
        draws = build_history(df)

        # Build panel dataset
        X, y = per_number_features(draws, lookback_short=10, lookback_long=50, decay=0.96)

        # Scale features (exclude the 'k' categorical id from scaling? Keep it; scaler will handle)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time-ordered split: last 15% as validation
        n = X_scaled.shape[0]
        split = int(n * 0.85)
        X_train, y_train = X_scaled[:split], y[:split]
        X_val,   y_val   = X_scaled[split:], y[split:]

        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_cache = models_dir / "model.h5"  # Use .h5 extension for TensorFlow models

        if model_cache.exists():
            try:
                model = tf.keras.models.load_model(str(model_cache))
            except Exception as e:
                logging.warning(f"Could not load cached model: {e}")
                model = None
            
            if model is None:
                model = build_model(X_train.shape[1])
                es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=100, batch_size=512, verbose=0, callbacks=[es])
                model.save(str(model_cache))  # Convert Path to string
        else:
            model = build_model(X_train.shape[1])
            es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=100, batch_size=512, verbose=0, callbacks=[es])
            model.save(str(model_cache))  # Convert Path to string

        # Calibrate with temperature scaling on validation set
        p_val = model.predict(X_val, verbose=0).ravel()
        T = fit_temperature(p_val, y_val)

        # Next-draw features
        X_next = next_draw_features(draws, lookback_short=10, lookback_long=50, decay=0.96)
        X_next_scaled = scaler.transform(X_next)

        p_next = model.predict(X_next_scaled, verbose=0).ravel()
        p_next = temperature_scale(p_next, T)

        # Sample 6 numbers
        combo = sample_six(p_next, enforce_constraints=True)
        combo_str = "-".join(f"{n:02d}" for n in sorted(combo))

        # Confidence score: mean calibrated prob of chosen numbers
        score = float(np.mean([p_next[n-1] for n in combo]))
        return combo_str, score

    except TimeoutError:
        return "Error: Generation timed out. Please try again.", 0.0
    except FileNotFoundError:
        return "Error: combinations.csv not found. Run csvcreation.py first.", 0.0
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", 0.0

# ----------------- GUI -----------------
# Add window centering and error handling for theme
def create_gui():
    try:
        # Initialize result queue for thread communication
        global result_queue
        result_queue = Queue()
        
        root = ThemedTk()
        try:
            root.set_theme("azure")
        except:
            logging.warning("Azure theme not available, using default theme")
            try:
                root.set_theme("breeze")  # Fallback theme
            except:
                logging.warning("Fallback theme not available, using system default")

        root.title("Ultra658 Deep Predictor")
        root.configure(bg='#f0f0f0')

        # Center window
        window_width = 480
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # Style configuration
        style = ttk.Style()
        style.configure('Modern.TButton', 
                       font=('Segoe UI', 10),
                       padding=10)
        style.configure('Title.TLabel',
                       font=('Segoe UI', 16, 'bold'),
                       background='#f0f0f0')
        style.configure('Result.TLabel',
                       font=('Segoe UI', 20, 'bold'),
                       background='#f0f0f0')
        style.configure('Notification.TLabel',
                       font=('Segoe UI', 10),
                       background='#f0f0f0',
                       padding=10)
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, 
                             text="Ultra658 Deep Predictor",
                             style='Title.TLabel')
        title_label.pack(pady=20)

        # Result frame
        result_frame = ttk.Frame(main_frame, padding="10")
        result_frame.pack(fill='x', pady=10)
        
        result_var = tk.StringVar(value="--:--:--:--:--:--")
        result_label = ttk.Label(result_frame, 
                                textvariable=result_var,
                                style='Result.TLabel')
        result_label.pack()

        # Confidence score
        score_var = tk.StringVar(value="Confidence: N/A")
        score_label = ttk.Label(main_frame, 
                             textvariable=score_var,
                             font=('Segoe UI', 12))
        score_label.pack(pady=5)

        # History frame
        history_frame = ttk.LabelFrame(main_frame, 
                                     text="Recent Combinations",
                                     padding="10")
        history_frame.pack(fill='x', pady=10)

        history = []
        history_listbox = tk.Listbox(history_frame,
                                    height=5,
                                    font=('Segoe UI', 12),
                                    selectmode='single',
                                    activestyle='none',
                                    bg='#ffffff',
                                    fg='#333333')
        history_listbox.pack(fill='x', padx=5)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=20)

        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill='x', pady=5)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=progress_var,
            maximum=100,
            mode='indeterminate'
        )
        
        def process_generation():
            try:
                progress_bar.pack(fill='x', pady=5)
                progress_bar.start(10)
                combo, score = generate_combination()
                result_queue.put((combo, score))
            except Exception as e:
                result_queue.put((f"Error: {str(e)}", 0.0))
            finally:
                progress_bar.stop()
                progress_bar.pack_forget()
        
        def check_queue():
            try:
                if not result_queue.empty():
                    combo, score = result_queue.get_nowait()
                    result_var.set(combo)
                    score_var.set(f"Confidence: {score:.4f}")
                    if "Error" not in combo:
                        history.append(combo)
                        history_listbox.delete(0, tk.END)
                        for h in history[-5:]:
                            history_listbox.insert(tk.END, h)
                    generate_btn.configure(state='normal')
                    root.config(cursor="")
                else:
                    # Check again in 100ms
                    root.after(100, check_queue)
            except Exception as e:
                logging.error(f"Queue check error: {e}")
                generate_btn.configure(state='normal')
                root.config(cursor="")

        def on_generate():
            try:
                generate_btn.configure(state='disabled')
                root.config(cursor="wait")
                root.update()
                
                # Start generation in separate thread
                Thread(target=process_generation, daemon=True).start()
                # Start checking for results
                root.after(100, check_queue)
                
            except Exception as e:
                logging.error(f"Generation error: {e}")
                generate_btn.configure(state='normal')
                root.config(cursor="")

        def copy_to_clipboard():
            text = result_var.get()
            if text == "--:--:--:--:--:--" or "Error" in text:
                show_notification("Warning", "Generate a combination first!")
            else:
                root.clipboard_clear()
                root.clipboard_append(text)
                show_notification("Success", "Combination copied!")

        def save_to_file():
            try:
                text = result_var.get()
                if text == "--:--:--:--:--:--" or "Error" in text:
                    show_notification("Warning", "Generate a combination first!")
                else:
                    # Create output directory
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Create a filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = output_dir / f"combinations_{timestamp}.txt"
                    
                    # Save with confidence score
                    with open(filename, "w") as f:
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Combination: {text}\n")
                        f.write(f"Confidence: {score_var.get().split(': ')[1]}\n")
                        
                    show_notification("Success", f"Saved to {filename.name}!")
            except Exception as e:
                logging.error(f"Error saving to file: {e}")
                show_notification("Error", "Failed to save combination")

        # Modern buttons
        generate_btn = ttk.Button(button_frame,
                                 text="Generate Numbers",
                                 style='Modern.TButton',
                                 command=on_generate)
        generate_btn.pack(fill='x', pady=5)

        copy_btn = ttk.Button(button_frame,
                             text="Copy to Clipboard",
                             style='Modern.TButton',
                             command=copy_to_clipboard)
        copy_btn.pack(fill='x', pady=5)

        save_btn = ttk.Button(button_frame,
                             text="Save to File",
                             style='Modern.TButton',
                             command=save_to_file)
        save_btn.pack(fill='x', pady=5)

        def on_closing():
            try:
                # Stop any running threads
                for thread in threading.enumerate():
                    if thread != threading.main_thread():
                        thread.daemon = True
        
                # Clear TensorFlow session
                tf.keras.backend.clear_session()
        
                # Save history with timestamp
                if history:
                    try:
                        # Create output directory
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        history_file = output_dir / f"history_{timestamp}.txt"
                        with open(history_file, "w") as f:
                            f.write(f"History saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            for h in history:
                                f.write(f"{h}\n")
                    except Exception as e:
                        logging.error(f"Error saving history: {e}")
            finally:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        def show_notification(title, message):
            notification = tk.Toplevel(root)
            notification.geometry("300x100")
            notification.title(title)
            notification.configure(bg='#f0f0f0')
            
            # Center notification
            notification.geometry(f"+{root.winfo_x() + 90}+{root.winfo_y() + 250}")
            
            ttk.Label(notification, 
                     text=message, 
                     padding=20,
                     font=('Segoe UI', 10)).pack()
                     
            notification.after(2000, notification.destroy)

        root.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to create GUI: {str(e)}")
        return

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Enabled memory growth for GPU: {gpu}")
                except RuntimeError as e:
                    logging.error(f"GPU memory config error: {e}")
        
        # Set thread count for parallel processing
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
        create_gui()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", str(e))
    finally:
        logging.info("Application closed")
