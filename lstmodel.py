import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# ------------ Feature Engineering ------------
def extract_features(df, window=5):
    """
    Extract features using last `window` draws.
    """
    features = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        sub_df = df.iloc[start:i+1]

        # Aggregate features across the window
        nums = []
        for combo in sub_df["Combinations"]:
            nums.extend([int(n) for n in combo.split("-")])
        nums = sorted(nums)

        # Basic stats
        total_sum = np.mean([sum([int(n) for n in combo.split("-")]) for combo in sub_df["Combinations"]])
        even_count = sum(1 for n in nums if n % 2 == 0)
        spread = max(nums) - min(nums) if nums else 0
        consec = sum(1 for j in range(len(nums) - 1) if nums[j+1] == nums[j] + 1)

        features.append([total_sum, even_count, spread, consec])
    return np.array(features)

# ------------ Model Training ------------
def train_model(X, y):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(58, activation='sigmoid')  # Independent probability per number
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model

# ------------ Sampling Method ------------
def sample_combination(probabilities, n_samples=6):
    """
    Direct probabilistic sampling based on model outputs.
    """
    probabilities = probabilities / probabilities.sum()  # Normalize
    chosen = np.random.choice(range(1, 59), size=n_samples, replace=False, p=probabilities)
    return sorted(chosen)

# ------------ Generate Combination ------------
def generate_combination():
    try:
        # Read CSV
        df = pd.read_csv("combinations.csv")

        # Extract features
        X = extract_features(df, window=5)

        # Build labels
        y = np.zeros((len(df), 58))
        for i, combo in enumerate(df["Combinations"]):
            nums = [int(n) for n in combo.split("-")]
            for n in nums:
                y[i, n-1] = 1  # Binary indicator

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = train_model(X_scaled, y)

        # Predict for next draw using last feature row
        latest_features = X_scaled[-1:].reshape(1, -1)
        probabilities = model.predict(latest_features, verbose=0)[0]

        # Sample new combination
        combo = sample_combination(probabilities)
        combo_str = "-".join(f"{num:02d}" for num in combo)
        score = sum(probabilities[n-1] for n in combo) / len(combo)
        return combo_str, score

    except FileNotFoundError:
        return "Error: combinations.csv not found. Run csvcreation.py first.", 0.0

# ------------ GUI ------------
def create_gui():
    root = tk.Tk()
    root.title("Ultra Lotto 6/58 Simulator")
    root.geometry("350x300")

    # Label
    label = tk.Label(root, text="Generate a Lotto combination!", font=("Arial", 12))
    label.pack(pady=10)

    # Combination display
    result_var = tk.StringVar()
    result_var.set("No combination yet")
    result_label = tk.Label(root, textvariable=result_var, font=("Arial", 14, "bold"))
    result_label.pack(pady=10)

    # Confidence score
    score_var = tk.StringVar()
    score_var.set("Confidence: N/A")
    score_label = tk.Label(root, textvariable=score_var, font=("Arial", 10))
    score_label.pack(pady=5)

    # History listbox
    history = []
    history_listbox = tk.Listbox(root, width=25, height=5, font=("Arial", 10))
    history_listbox.pack(pady=5)

    # Generate button
    def on_generate():
        combo, score = generate_combination()
        result_var.set(combo)
        score_var.set(f"Confidence: {score:.4f}")
        if "Error" not in combo:
            history.append(combo)
            history_listbox.delete(0, tk.END)
            for h in history[-5:]:
                history_listbox.insert(tk.END, h)

    button = tk.Button(root, text="Generate Combination", command=on_generate, font=("Arial", 10))
    button.pack(pady=5)

    # Copy to clipboard
    def copy_to_clipboard():
        if result_var.get() == "No combination yet" or "Error" in result_var.get():
            messagebox.showwarning("Warning", "Generate a combination first!")
        else:
            root.clipboard_clear()
            root.clipboard_append(result_var.get())
            messagebox.showinfo("Success", "Combination copied to clipboard!")

    copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard, font=("Arial", 10))
    copy_button.pack(pady=5)

    # Save to file
    def save_to_file():
        if result_var.get() == "No combination yet" or "Error" in result_var.get():
            messagebox.showwarning("Warning", "Generate a combination first!")
        else:
            with open("generated_combinations.txt", "a") as f:
                f.write(result_var.get() + "\n")
            messagebox.showinfo("Success", "Combination saved to generated_combinations.txt!")

    save_button = tk.Button(root, text="Save to File", command=save_to_file, font=("Arial", 10))
    save_button.pack(pady=5)

    root.mainloop()

# ------------ Main ------------
if __name__ == "__main__":
    create_gui()
