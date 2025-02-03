import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TimeSeriesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Viewer")
        self.data = None

        # Create GUI elements
        self.load_button = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.load_button.pack()

        self.column_label = tk.Label(root, text="Select Column:")
        self.column_label.pack()

        self.column_combobox = ttk.Combobox(root, state="readonly")
        self.column_combobox.pack()

        self.date_range_label = tk.Label(root, text="Select Date Range (YYYY-MM-DD HH:MM:SS):")
        self.date_range_label.pack()

        self.start_date_entry = tk.Entry(root)
        self.start_date_entry.pack()
        self.start_date_entry.insert(0, "Start Date-Time")

        self.end_date_entry = tk.Entry(root)
        self.end_date_entry.pack()
        self.end_date_entry.insert(0, "End Date-Time")

        self.plot_button = tk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.pack()

        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path, parse_dates=["Time"])
            self.data.set_index("Time", inplace=True)
            self.column_combobox["values"] = self.data.columns.tolist()

    def plot_data(self):
        if self.data is not None:
            selected_column = self.column_combobox.get()
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()

            if selected_column and start_date and end_date:
                try:
                    filtered_data = self.data.loc[start_date:end_date, selected_column]
                    self.ax.clear()
                    self.ax.scatter(filtered_data.index, filtered_data.values, label=selected_column, s = 1)
                    self.ax.plot(filtered_data.index, filtered_data.values, label=selected_column, alpha = 0.2)
                    self.ax.set_title(f"Time Series: {selected_column}")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel(selected_column)
                    self.ax.legend()
                    self.canvas.draw()
                except Exception as e:
                    tk.messagebox.showerror("Error", f"Failed to plot data: {e}")
            else:
                tk.messagebox.showwarning("Input Error", "Please select a column and specify a valid date range.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesApp(root)
    root.mainloop()

