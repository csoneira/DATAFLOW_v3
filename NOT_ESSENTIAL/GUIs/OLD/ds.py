import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization Tool")
        self.data_dict = {}  # Dictionary to store data from multiple CSVs
        self.mode = "Time Series"  # Default mode

        # Mode Selection
        self.mode_label = tk.Label(root, text="Select Mode:")
        self.mode_label.pack()
        self.mode_var = tk.StringVar(value="Time Series")
        self.mode_menu = ttk.Combobox(root, textvariable=self.mode_var, values=["Time Series", "Histogram"], state="readonly")
        self.mode_menu.pack()
        self.mode_menu.bind("<<ComboboxSelected>>", self.change_mode)

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

        # Axis Selection Label
        self.axis_label = tk.Label(root, text="Select Axis for Columns:")
        self.axis_label.pack()

        # Axis Selection Combobox
        self.axis_var = tk.StringVar(value="Primary")
        self.axis_menu = ttk.Combobox(root, textvariable=self.axis_var, values=["Primary", "Secondary"], state="readonly")
        self.axis_menu.pack()

        # Date Range Label (for Time Series)
        self.date_range_label = tk.Label(root, text="Select Date Range (YYYY-MM-DD HH:MM:SS):")
        self.date_range_label.pack()

        # Start Date Entry (for Time Series)
        self.start_date_entry = tk.Entry(root)
        self.start_date_entry.pack()
        self.start_date_entry.insert(0, "Start Date-Time")

        # End Date Entry (for Time Series)
        self.end_date_entry = tk.Entry(root)
        self.end_date_entry.pack()
        self.end_date_entry.insert(0, "End Date-Time")

        # Limits Label (for Histogram)
        self.limits_label = tk.Label(root, text="Select Limits:")
        self.limits_label.pack()

        # Left Limit Entry (for Histogram)
        self.left_entry = tk.Entry(root)
        self.left_entry.pack()
        self.left_entry.insert(0, "Left Limit")

        # Right Limit Entry (for Histogram)
        self.right_entry = tk.Entry(root)
        self.right_entry.pack()
        self.right_entry.insert(0, "Right Limit")

        # Plot Button
        self.plot_button = tk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.pack()

        # Matplotlib Figure
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()  # Secondary axis
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

    def change_mode(self, event):
        """Changes the mode between Time Series and Histogram."""
        self.mode = self.mode_var.get()
        if self.mode == "Time Series":
            self.date_range_label.pack()
            self.start_date_entry.pack()
            self.end_date_entry.pack()
            self.limits_label.pack_forget()
            self.left_entry.pack_forget()
            self.right_entry.pack_forget()
        else:
            self.date_range_label.pack_forget()
            self.start_date_entry.pack_forget()
            self.end_date_entry.pack_forget()
            self.limits_label.pack()
            self.left_entry.pack()
            self.right_entry.pack()

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
        """Plots selected columns from the selected file(s) over the same x-axis."""
        if self.mode == "Time Series":
            self.plot_time_series()
        else:
            self.plot_histogram()

    def plot_time_series(self):
        """Plots time series data."""
        selected_file_indices = self.file_listbox.curselection()
        if selected_file_indices:
            selected_files = [list(self.data_dict.keys())[i] for i in selected_file_indices]
            selected_indices = self.column_listbox.curselection()
            selected_columns = [self.column_listbox.get(i) for i in selected_indices]
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()

            if selected_columns and start_date and end_date:
                try:
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    
                    # Clear previous plot
                    self.ax.clear()
                    self.ax2.clear()

                    for file_path in selected_files:
                        filtered_data = self.data_dict[file_path].loc[start_date:end_date, selected_columns]
                        for column in selected_columns:
                            axis = self.ax if self.axis_var.get() == "Primary" else self.ax2
                            axis.scatter(filtered_data.index, filtered_data[column], label=f"{file_path.split('/')[-1]}: {column}", s=1)
                            axis.plot(filtered_data.index, filtered_data[column], alpha=0.5)
                    
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

    def plot_histogram(self):
        """Plots histogram data."""
        selected_file_indices = self.file_listbox.curselection()
        if selected_file_indices:
            selected_files = [list(self.data_dict.keys())[i] for i in selected_file_indices]
            selected_indices = self.column_listbox.curselection()
            selected_columns = [self.column_listbox.get(i) for i in selected_indices]

            if selected_columns:
                try:
                    left_limit = float(self.left_entry.get())
                    right_limit = float(self.right_entry.get())
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numeric limits!")
                    return

                # Clear previous plot
                self.ax.clear()

                for file_path in selected_files:
                    for column in selected_columns:
                        filtered_data = self.data_dict[file_path][column].dropna()  # Remove NaNs
                        filtered_data = filtered_data[filtered_data != 0]  # Remove 0s
                        filtered_data = filtered_data[(filtered_data >= left_limit) & (filtered_data <= right_limit)]  # Apply limits

                        if not filtered_data.empty:
                            self.ax.hist(filtered_data, bins=30, edgecolor='black', alpha=0.75, label=f"{file_path.split('/')[-1]}: {column}")

                self.ax.set_xlabel("Values")
                self.ax.set_ylabel("Frequency")
                self.ax.set_title("Filtered Histogram")
                self.ax.legend()
                self.canvas.draw()
            else:
                messagebox.showwarning("Input Error", "Please select at least one column.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizationApp(root)
    root.mainloop()