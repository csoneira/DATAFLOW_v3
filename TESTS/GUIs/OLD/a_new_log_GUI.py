import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TimeSeriesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi CSV Time Series Viewer")
        self.data_dict = {}  # Dictionary to store data from multiple CSVs

        # Load CSV Button
        self.load_button = tk.Button(root, text="Load CSV(s)", command=self.load_csvs)
        self.load_button.pack()

        # File List Label
        self.file_list_label = tk.Label(root, text="Loaded Files:")
        self.file_list_label.pack()

        # Listbox for Files
        self.file_listbox = tk.Listbox(root, height=6, exportselection=0)
        self.file_listbox.pack()
        self.file_listbox.bind("<<ListboxSelect>>", self.populate_columns)

        # Column Selection Label
        self.column_label = tk.Label(root, text="Select Columns:")
        self.column_label.pack()

        # Multi-Select Listbox for Columns
        self.column_listbox = tk.Listbox(root, selectmode="multiple", height=6, exportselection=0)
        self.column_listbox.pack()

        # Date Range Label
        self.date_range_label = tk.Label(root, text="Select Date Range (YYYY-MM-DD HH:MM:SS):")
        self.date_range_label.pack()

        # Start Date Entry
        self.start_date_entry = tk.Entry(root)
        self.start_date_entry.pack()
        self.start_date_entry.insert(0, "Start Date-Time")

        # End Date Entry
        self.end_date_entry = tk.Entry(root)
        self.end_date_entry.pack()
        self.end_date_entry.insert(0, "End Date-Time")

        # Plot Button
        self.plot_button = tk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.pack()

        # Matplotlib Figure
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

    def load_csvs(self):
        """Loads multiple CSV files and allows column selection from each."""
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        
        if file_paths:
            for file_path in file_paths:
                try:
                    data = pd.read_csv(file_path, parse_dates=["Time"])
                    data.set_index("Time", inplace=True)
                    self.data_dict[file_path] = data
                    self.file_listbox.insert(tk.END, file_path.split("/")[-1])  # Show filename
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {file_path}: {e}")

    def populate_columns(self, event):
        """Populates the column listbox based on the selected file."""
        selected_file_index = self.file_listbox.curselection()
        if selected_file_index:
            selected_file = list(self.data_dict.keys())[selected_file_index[0]]
            self.column_listbox.delete(0, tk.END)  # Clear previous list
            for col in self.data_dict[selected_file].columns:
                self.column_listbox.insert(tk.END, col)  # Populate with column names

    def plot_data(self):
        """Plots selected columns from the selected file over the same x-axis."""
        selected_file_index = self.file_listbox.curselection()
        if selected_file_index:
            selected_file = list(self.data_dict.keys())[selected_file_index[0]]
            selected_indices = self.column_listbox.curselection()
            selected_columns = [self.column_listbox.get(i) for i in selected_indices]
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()

            if selected_columns and start_date and end_date:
                try:
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    
                    filtered_data = self.data_dict[selected_file].loc[start_date:end_date, selected_columns]
                    
                    # Clear previous plot
                    self.ax.clear()

                    # Plot each selected column
                    for column in selected_columns:
                        self.ax.scatter(filtered_data.index, filtered_data[column], label=column, s=1)
                        self.ax.plot(filtered_data.index, filtered_data[column], alpha=0.5)
                    
                    self.ax.set_xlim(start_date, end_date)
                    self.ax.set_title(f"Time Series: {', '.join(selected_columns)}")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Values")
                    self.ax.legend()
                    self.canvas.draw()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to plot data: {e}")
            else:
                messagebox.showwarning("Input Error", "Please select at least one column and specify a valid date range.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesApp(root)
    root.mainloop()
