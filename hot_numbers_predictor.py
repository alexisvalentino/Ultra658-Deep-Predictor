import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hot_numbers_predictor.log'
)

class HotNumbersPredictor:
    def __init__(self):
        self.hot_numbers = [13, 6, 38, 22, 40, 33]  # Top 6 most frequent
        self.hot_numbers_set = set(self.hot_numbers)
        self.model = None
        self.scaler = None
        
    def analyze_hot_numbers_performance(self, df):
        """Analyze how often hot numbers appear in combinations"""
        hot_numbers_per_combo = []
        
        for combo_str in df["Combinations"]:
            numbers = set(int(n) for n in combo_str.split("-"))
            hot_count = len(numbers.intersection(self.hot_numbers_set))
            hot_numbers_per_combo.append(hot_count)
        
        hot_counts = Counter(hot_numbers_per_combo)
        total_combos = len(hot_numbers_per_combo)
        
        print("\n=== HOT NUMBERS PERFORMANCE ANALYSIS ===")
        print(f"Total combinations: {total_combos}")
        print(f"Hot numbers: {sorted(self.hot_numbers)}")
        
        for count in range(7):
            if count in hot_counts:
                percentage = (hot_counts[count] / total_combos) * 100
                print(f"{count} hot numbers: {hot_counts[count]} times ({percentage:.2f}%)")
        
        # Calculate probability of getting at least 3 hot numbers
        at_least_3 = sum(hot_counts[i] for i in range(3, 7))
        prob_at_least_3 = (at_least_3 / total_combos) * 100
        print(f"\nProbability of getting AT LEAST 3 hot numbers: {prob_at_least_3:.2f}%")
        
        return hot_counts
    
    def build_hot_numbers_features(self, draws, lookback_short=10, lookback_long=50, decay=0.96):
        """Build features specifically for predicting hot numbers appearance"""
        rows = []
        targets = []
        T = len(draws)
        
        for t in range(T):
            if t == 0:
                continue
                
            current_draw = draws[t]
            previous_draws = draws[:t]
            
            # Feature 1: How many hot numbers appeared in the last draw
            last_hot_count = len(set(previous_draws[-1]).intersection(self.hot_numbers_set))
            
            # Feature 2: Average hot numbers per draw in short window
            short_window = previous_draws[-lookback_short:] if len(previous_draws) >= lookback_short else previous_draws
            short_hot_avg = np.mean([len(set(draw).intersection(self.hot_numbers_set)) for draw in short_window])
            
            # Feature 3: Average hot numbers per draw in long window
            long_window = previous_draws[-lookback_long:] if len(previous_draws) >= lookback_long else previous_draws
            long_hot_avg = np.mean([len(set(draw).intersection(self.hot_numbers_set)) for draw in long_window])
            
            # Feature 4: Momentum (short vs long term)
            momentum = short_hot_avg - long_hot_avg
            
            # Feature 5: Time since last high hot number count (3+)
            time_since_high = 0
            for i in range(t-1, -1, -1):
                hot_count = len(set(previous_draws[i]).intersection(self.hot_numbers_set))
                if hot_count >= 3:
                    time_since_high = t - 1 - i
                    break
                time_since_high += 1
            
            # Feature 6: Decayed hot numbers frequency
            decayed_freq = 0.0
            for i in range(t-1, -1, -1):
                hot_count = len(set(previous_draws[i]).intersection(self.hot_numbers_set))
                decayed_freq = decay * decayed_freq + hot_count
            
            # Feature 7: Hot numbers streak (consecutive draws with 2+ hot numbers)
            streak = 0
            for i in range(t-1, -1, -1):
                hot_count = len(set(previous_draws[i]).intersection(self.hot_numbers_set))
                if hot_count >= 2:
                    streak += 1
                else:
                    break
            
            features = [
                last_hot_count,
                short_hot_avg,
                long_hot_avg,
                momentum,
                time_since_high,
                decayed_freq,
                streak
            ]
            
            # Target: Will next draw have 3+ hot numbers?
            next_hot_count = len(set(current_draw).intersection(self.hot_numbers_set))
            target = 1 if next_hot_count >= 3 else 0
            
            rows.append(features)
            targets.append(target)
        
        return np.array(rows, dtype=float), np.array(targets, dtype=float)
    
    def build_hot_numbers_model(self, input_dim):
        """Build a model specifically for predicting hot numbers appearance"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, df):
        """Train the hot numbers prediction model"""
        try:
            draws = []
            for combo_str in df["Combinations"]:
                draws.append(set(int(n) for n in combo_str.split("-")))
            
            # Build features for hot numbers prediction
            X, y = self.build_hot_numbers_features(draws)
            
            if X is None or len(X) == 0:
                raise ValueError("No features generated")
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            split = int(len(X_scaled) * 0.85)
            X_train, y_train = X_scaled[:split], y[:split]
            X_val, y_val = X_scaled[split:], y[split:]
            
            # Build and train model
            self.model = self.build_hot_numbers_model(X_train.shape[1])
            
            es = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                verbose=0,
                callbacks=[es]
            )
            
            # Evaluate model
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            print(f"\nModel trained successfully!")
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            print(f"Training failed: {str(e)}")
            return False
    
    def predict_next_hot_numbers_probability(self, df):
        """Predict probability of getting 3+ hot numbers in next draw"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            draws = []
            for combo_str in df["Combinations"]:
                draws.append(set(int(n) for n in combo_str.split("-")))
            
            # Build features for next draw
            X_next = self.build_hot_numbers_features(draws)[0][-1:]  # Last row
            
            # Scale features
            X_next_scaled = self.scaler.transform(X_next)
            
            # Predict probability
            prob = self.model.predict(X_next_scaled, verbose=0)[0][0]
            
            return prob
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5
    
    def generate_smart_combination(self, df, target_hot_count=3):
        """Generate combination with high probability of containing target hot numbers"""
        try:
            # Get current hot numbers probability
            hot_prob = self.predict_next_hot_numbers_probability(df)
            
            print(f"\n=== SMART COMBINATION GENERATION ===")
            print(f"Probability of 3+ hot numbers in next draw: {hot_prob:.4f}")
            
            # Strategy: Ensure we get target_hot_count hot numbers
            hot_numbers_to_include = random.sample(self.hot_numbers, target_hot_count)
            remaining_slots = 6 - target_hot_count
            
            # Fill remaining slots with other numbers (avoiding too many hot numbers)
            other_numbers = [n for n in range(1, 59) if n not in self.hot_numbers]
            remaining_numbers = random.sample(other_numbers, remaining_slots)
            
            # Combine and sort
            final_combination = sorted(hot_numbers_to_include + remaining_numbers)
            combination_str = "-".join(f"{n:02d}" for n in final_combination)
            
            # Calculate actual hot numbers in this combination
            actual_hot_count = len(set(final_combination).intersection(self.hot_numbers_set))
            
            print(f"Target hot numbers: {target_hot_count}")
            print(f"Actual hot numbers: {actual_hot_count}")
            print(f"Combination: {combination_str}")
            print(f"Hot numbers included: {sorted(hot_numbers_to_include)}")
            
            return combination_str, actual_hot_count, hot_prob
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "Error generating combination", 0, 0.0

def create_hot_numbers_gui():
    """Create GUI for hot numbers predictor"""
    try:
        root = ThemedTk()
        try:
            root.set_theme("azure")
        except:
            try:
                root.set_theme("breeze")
            except:
                pass
        
        root.title("Hot Numbers Predictor")
        root.configure(bg='#f0f0f0')
        
        # Center window
        window_width = 600
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                             text="Hot Numbers Predictor",
                             font=('Segoe UI', 18, 'bold'))
        title_label.pack(pady=20)
        
        # Hot numbers display
        hot_frame = ttk.LabelFrame(main_frame, text="Top 6 Most Frequent Numbers", padding="10")
        hot_frame.pack(fill='x', pady=10)
        
        hot_numbers = [13, 6, 38, 22, 40, 33]
        hot_text = " ".join(f"{n:02d}" for n in hot_numbers)
        hot_label = ttk.Label(hot_frame, text=hot_text, font=('Segoe UI', 14))
        hot_label.pack()
        
        # Analysis button
        analyze_btn = ttk.Button(main_frame, text="Analyze Hot Numbers Performance", 
                               command=lambda: analyze_performance())
        analyze_btn.pack(fill='x', pady=10)
        
        # Train button
        train_btn = ttk.Button(main_frame, text="Train Hot Numbers Model", 
                             command=lambda: train_model())
        train_btn.pack(fill='x', pady=5)
        
        # Generate button
        generate_btn = ttk.Button(main_frame, text="Generate Smart Combination", 
                                command=lambda: generate_combination())
        generate_btn.pack(fill='x', pady=5)
        
        # Results frame
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.pack(fill='both', expand=True, pady=10)
        
        result_text = tk.Text(result_frame, height=15, font=('Consolas', 10))
        result_text.pack(fill='both', expand=True)
        
        # Initialize predictor
        predictor = HotNumbersPredictor()
        
        def analyze_performance():
            try:
                df = pd.read_csv("combinations.csv")
                hot_counts = predictor.analyze_hot_numbers_performance(df)
                
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "=== HOT NUMBERS PERFORMANCE ANALYSIS ===\n\n")
                
                for count in range(7):
                    if count in hot_counts:
                        percentage = (hot_counts[count] / len(df)) * 100
                        result_text.insert(tk.END, f"{count} hot numbers: {hot_counts[count]} times ({percentage:.2f}%)\n")
                
                at_least_3 = sum(hot_counts[i] for i in range(3, 7))
                prob_at_least_3 = (at_least_3 / len(df)) * 100
                result_text.insert(tk.END, f"\nProbability of getting AT LEAST 3 hot numbers: {prob_at_least_3:.2f}%\n")
                
            except Exception as e:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Error: {str(e)}")
        
        def train_model():
            try:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "Training model...\n")
                root.update()
                
                df = pd.read_csv("combinations.csv")
                success = predictor.train_model(df)
                
                if success:
                    result_text.insert(tk.END, "Model trained successfully!\n")
                    result_text.insert(tk.END, "You can now generate smart combinations.\n")
                else:
                    result_text.insert(tk.END, "Model training failed.\n")
                    
            except Exception as e:
                result_text.insert(tk.END, f"Error: {str(e)}")
        
        def generate_combination():
            try:
                if predictor.model is None:
                    result_text.insert(tk.END, "Please train the model first!\n")
                    return
                
                df = pd.read_csv("combinations.csv")
                combo, hot_count, prob = predictor.generate_smart_combination(df)
                
                result_text.insert(tk.END, f"\n=== GENERATED COMBINATION ===\n")
                result_text.insert(tk.END, f"Combination: {combo}\n")
                result_text.insert(tk.END, f"Hot numbers included: {hot_count}\n")
                result_text.insert(tk.END, f"Probability of 3+ hot numbers: {prob:.4f}\n")
                
            except Exception as e:
                result_text.insert(tk.END, f"Error: {str(e)}")
        
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create GUI: {str(e)}")

if __name__ == "__main__":
    try:
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        create_hot_numbers_gui()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", str(e))
