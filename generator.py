import pandas as pd
import random
import tkinter as tk
from tkinter import messagebox

# Generate a combination based on number frequency
def generate_combination():
    try:
        # Read the CSV
        df = pd.read_csv("combinations.csv")
        # Split combinations into individual numbers
        numbers = []
        for combo in df["Combinations"]:
            nums = [int(n) for n in combo.split("-")]
            numbers.extend(nums)
        
        # Calculate frequency of each number (1 to 58)
        freq = {}
        for num in range(1, 59):
            freq[num] = numbers.count(num)
        
        # Sort numbers by frequency (descending)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 6 numbers based on frequency, with randomness for ties
        top_numbers = []
        i = 0
        while len(top_numbers) < 6 and i < len(sorted_freq):
            current_freq = sorted_freq[i][1]
            # Get all numbers with the same frequency
            same_freq = [num for num, f in sorted_freq if f == current_freq]
            # Shuffle and pick needed numbers
            random.shuffle(same_freq)
            top_numbers.extend(same_freq[:6 - len(top_numbers)])
            i += len(same_freq)
        
        # Ensure exactly 6 numbers, sorted for consistency
        top_numbers = sorted(top_numbers[:6])
        # Format as string
        return "-".join(f"{num:02d}" for num in top_numbers)
    
    except FileNotFoundError:
        return "Error: combinations.csv not found. Run csvcreation.py first."

# Create GUI
def create_gui():
    root = tk.Tk()
    root.title("Ultra Lotto 6/58 Simulator")
    root.geometry("300x250")
    
    # Label
    label = tk.Label(root, text="Click to generate a Lotto combination!", font=("Arial", 12))
    label.pack(pady=10)
    
    # Combination display
    result_var = tk.StringVar()
    result_var.set("No combination yet")
    result_label = tk.Label(root, textvariable=result_var, font=("Arial", 14, "bold"))
    result_label.pack(pady=10)
    
    # History listbox
    history = []
    history_listbox = tk.Listbox(root, width=20, height=5, font=("Arial", 10))
    history_listbox.pack(pady=5)
    
    # Generate button
    def on_generate():
        combo = generate_combination()
        result_var.set(combo)
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
    
    # Run the GUI
    root.mainloop()

# Main execution
if __name__ == "__main__":
    create_gui()