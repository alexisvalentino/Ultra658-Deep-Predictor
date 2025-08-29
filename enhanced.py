import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Feature engineering
def extract_features(df):
    features = []
    for combo in df["Combinations"]:
        nums = sorted([int(n) for n in combo.split("-")])
        # Sum of numbers
        total_sum = sum(nums)
        # Even/odd count
        even_count = len([n for n in nums if n % 2 == 0])
        # Spread
        spread = max(nums) - min(nums)
        # Consecutive pairs
        consec = sum(1 for i in range(5) if nums[i+1] == nums[i] + 1)
        # Positional stats
        features.append([total_sum, even_count, spread, consec] + nums)
    return np.array(features)

# Train a simple neural network
def train_model(X, y):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(58, activation='softmax')  # Probabilities for numbers 1-58
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    return model

# MCMC sampling for combination generation
def mcmc_sample(probabilities, n_samples=6, steps=1000, temp=1.0):
    current = sorted(random.sample(range(1, 59), n_samples))
    best = current[:]
    best_score = sum(probabilities[n-1] for n in current) / temp
    
    for _ in range(steps):
        # Propose a new combination by swapping one number
        new_combo = current[:]
        idx = random.randint(0, 5)
        new_num = random.randint(1, 58)
        while new_num in new_combo:
            new_num = random.randint(1, 58)
        new_combo[idx] = new_num
        new_combo = sorted(new_combo)
        
        # Calculate acceptance probability
        new_score = sum(probabilities[n-1] for n in new_combo) / temp
        if new_score > best_score or random.random() < np.exp((new_score - best_score) / temp):
            current = new_combo
            if new_score > best_score:
                best = current[:]
                best_score = new_score
    
    return best, best_score

# Generate a combination
def generate_combination():
    try:
        # Read the CSV
        df = pd.read_csv("combinations.csv")
        
        # Extract features
        X = extract_features(df)
        # Create pseudo-labels (frequency-based)
        numbers = []
        for combo in df["Combinations"]:
            nums = [int(n) for n in combo.split("-")]
            numbers.extend(nums)
        freq = np.array([numbers.count(i) for i in range(1, 59)])
        freq = freq / freq.sum()  # Normalize to probabilities
        
        # Prepare data for neural network
        X_features = X[:, :4]  # Use sum, even_count, spread, consec
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        y = np.zeros((len(df), 58))
        for i, combo in enumerate(df["Combinations"]):
            nums = [int(n) for n in combo.split("-")]
            for n in nums:
                y[i, n-1] = 1 / 6  # Distribute probability across numbers
        
        # Train model
        model = train_model(X_scaled, y)
        
        # Predict probabilities for the next combination
        latest_features = X_scaled[-1:]  # Use features from the last draw
        probabilities = model.predict(latest_features, verbose=0)[0]
        
        # Adjust probabilities with frequency-based bias
        probabilities = 0.7 * probabilities + 0.3 * freq
        
        # Sample combination using MCMC
        combo, score = mcmc_sample(probabilities)
        combo_str = "-".join(f"{num:02d}" for num in combo)
        return combo_str, score
    
    except FileNotFoundError:
        return "Error: combinations.csv not found. Run csvcreation.py first.", 0.0

# Create GUI
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
            for h in history[-5:]:  # Show last 5 combinations
                history_listbox.insert(tk.END, h)
    
    button = tk.Button(root, text="Generate Combination", command=on_generate, font=("Arial", 10))
    button.pack(pady=5)
    
    # Copy to clipboard button
    def copy_to_clipboard():
        if result_var.get() == "No combination yet" or "Error" in result_var.get():
            messagebox.showwarning("Warning", "Generate a combination first!")
        else:
            root.clipboard_clear()
            root.clipboard_append(result_var.get())
            messagebox.showinfo("Success", "Combination copied to clipboard!")
    
    copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard, font=("Arial", 10))
    copy_button.pack(pady=5)
    
    # Save to file button
    def save_to_file():
        if result_var.get() == "No combination yet" or "Error" in result_var.get():
            messagebox.showwarning("Warning", "Generate a combination first!")
        else:
            with open("generated_combinations.txt", "a") as f:
                f.write(result_var.get() + "\n")
            messagebox.showinfo("Success", "Combination saved to generated_combinations.txt!")
    
    save_button = tk.Button(root, text="Save to File", command=save_to_file, font=("Arial", 10))
    save_button.pack(pady=5)
    
    # Run the GUI
    root.mainloop()

# Main execution
if __name__ == "__main__":
    create_gui()