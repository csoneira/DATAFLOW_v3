import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt

class CSVHistogramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Histogram Filter")

        # File Selection
        self.btn_load = tk.Button(root, text="Load Data File", command=self.load_csv)
        self.btn_load.pack(pady=5)

        # Dropdown for Column Selection
        self.column_label = tk.Label(root, text="Select Column:")
        self.column_label.pack()
        self.column_select = ttk.Combobox(root, state="readonly")
        self.column_select.pack(pady=5)

        # Entry for Left Limit
        self.left_label = tk.Label(root, text="Left Limit:")
        self.left_label.pack()
        self.left_entry = tk.Entry(root)
        self.left_entry.pack(pady=5)

        # Entry for Right Limit
        self.right_label = tk.Label(root, text="Right Limit:")
        self.right_label.pack()
        self.right_entry = tk.Entry(root)
        self.right_entry.pack(pady=5)

        # Process Button
        self.btn_process = tk.Button(root, text="Plot Histogram", command=self.process_data)
        self.btn_process.pack(pady=5)

        # Save Button
        self.btn_save = tk.Button(root, text="Save Histogram", command=self.save_histogram, state=tk.DISABLED)
        self.btn_save.pack(pady=5)

        self.df = None
        self.filtered_data = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(title="Select a Data File")
        if not file_path:
            return  # User cancelled file selection

        try:
            # Attempt to read the file as a CSV
            self.df = pd.read_csv(file_path)
            
            # Update column dropdown
            self.column_select["values"] = list(self.df.columns)
            self.column_select.current(0)  # Select first column by default
            messagebox.showinfo("Success", f"File loaded successfully: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file as CSV:\n{e}")

    def process_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No data file loaded!")
            return
        
        selected_column = self.column_select.get()
        if selected_column not in self.df.columns:
            messagebox.showerror("Error", "Invalid column selection!")
            return

        try:
            left_limit = float(self.left_entry.get())
            right_limit = float(self.right_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric limits!")
            return

        # Filtering the column
        self.filtered_data = self.df[selected_column].dropna()  # Remove NaNs
        self.filtered_data = self.filtered_data[self.filtered_data != 0]  # Remove 0s
        self.filtered_data = self.filtered_data[(self.filtered_data >= left_limit) & (self.filtered_data <= right_limit)]  # Apply limits

        if self.filtered_data.empty:
            messagebox.showwarning("Warning", "No data left after filtering!")
            return

        # Plot Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(self.filtered_data, bins=30, edgecolor='black', alpha=0.75)
        plt.xlabel(selected_column)
        plt.ylabel("Frequency")
        plt.title("Filtered Histogram")
        plt.grid(True)
        plt.show()

        self.btn_save.config(state=tk.NORMAL)

    def save_histogram(self):
        if self.filtered_data is None or self.filtered_data.empty:
            messagebox.showerror("Error", "No data to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"),
                                                            ("All Files", "*.*")])
        if file_path:
            plt.figure(figsize=(6, 4))
            plt.hist(self.filtered_data, bins=30, edgecolor='black', alpha=0.75)
            plt.xlabel(self.column_select.get())
            plt.ylabel("Frequency")
            plt.title("Filtered Histogram")
            plt.grid(True)
            plt.savefig(file_path)
            messagebox.showinfo("Success", f"Histogram saved as {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVHistogramGUI(root)
    root.mainloop()
