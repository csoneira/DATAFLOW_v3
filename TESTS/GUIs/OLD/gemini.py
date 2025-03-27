import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualizer")

        self.data_dict = {}
        self.plot_mode = "time_series"  # Default mode

        # Mode Selection
        self.mode_var = tk.StringVar(value="time_series")
        self.mode_menu = tk.OptionMenu(root, self.mode_var, "time_series", "histogram", command=self.switch_mode)
        self.mode_menu.pack()

        # File Loading
        self.load_button = tk.Button(root, text="Load CSV(s)", command=self.load_csvs)
        self.load_button.pack()
        self.file_listbox = tk.Listbox(root, height=6, exportselection=0)
        self.file_listbox.pack()
        self.file_listbox.bind("<<ListboxSelect>>", self.populate_columns)

        # Column Selection (Multi-select)
        self.column_label = tk.Label(root, text="Select Columns:")
        self.column_label.pack()
        self.column_listbox = tk.Listbox(root, selectmode="multiple", height=6, exportselection=0)
        self.column_listbox.pack()

        # Axis Selection (Dictionary to store axis for each column)
        self.column_axis = {}  # {file_path: {column: axis_number}}
        self.axis_options = ["shared"]  # Start with one shared axis
        self.axis_menu = None # Will be created dynamically
        self.add_axis_button = tk.Button(root, text="Add Axis", command=self.add_new_axis)
        self.add_axis_button.pack()

        # Date Range (Only for Time Series)
        self.date_range_frame = tk.Frame(root) # To hide/show in time series mode
        self.date_range_label = tk.Label(self.date_range_frame, text="Select Date Range (YYYY-MM-DD HH:MM:SS):")
        self.date_range_label.pack()
        self.start_date_entry = tk.Entry(self.date_range_frame)
        self.start_date_entry.pack()
        self.start_date_entry.insert(0, "Start Date-Time")
        self.end_date_entry = tk.Entry(self.date_range_frame)
        self.end_date_entry.pack()
        self.end_date_entry.insert(0, "End Date-Time")
        self.date_range_frame.pack()

        # Plot Button
        self.plot_button = tk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.pack()

        # Matplotlib Figure
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

    def switch_mode(self, mode):
        self.plot_mode = mode
        if mode == "time_series":
            self.date_range_frame.pack()  # Show date range options
            self.add_axis_button.pack()
        else:
            self.date_range_frame.pack_forget() # Hide date range options
            self.add_axis_button.pack_forget()

    def load_csvs(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            for file_path in file_paths:
                try:
                    data = pd.read_csv(file_path, parse_dates=["Time"] if self.plot_mode == "time_series" else None)
                    if self.plot_mode == "time_series":
                        data.set_index("Time", inplace=True)

                    self.data_dict[file_path] = data
                    self.file_listbox.insert(tk.END, file_path.split("/")[-1])
                    self.column_axis[file_path] = {} # Initialize axis dictionary for the file
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {file_path}: {e}")

    def populate_columns(self, event):
        selected_file_indices = self.file_listbox.curselection()
        self.column_listbox.delete(0, tk.END)
        if selected_file_indices:
            for index in selected_file_indices:
                selected_file = list(self.data_dict.keys())[index]
                for col in self.data_dict[selected_file].columns:
                    self.column_listbox.insert(tk.END, f"{selected_file.split('/')[-1]}: {col}") # Include filename
                    if col not in self.column_axis[selected_file]:
                        self.column_axis[selected_file][col] = "shared"  # Default axis
            self.update_axis_menu() # Update axis menu when columns are populated


    def update_axis_menu(self):
        if self.axis_menu:
            self.axis_menu.destroy()  # Destroy previous menu
        self.axis_menu = tk.Menubutton(self.root, text="Select Axis", relief=tk.RAISED)
        self.axis_menu.menu = tk.Menu(self.axis_menu, tearoff=0)
        self.axis_menu["menu"] = self.axis_menu.menu

        for file_path, columns in self.column_axis.items():
            for col in columns:
                var = tk.StringVar(value=self.column_axis[file_path][col]) # Get current value
                file_name = file_path.split("/")[-1]
                menu_label = f"{file_name}: {col}"
                col_menu = tk.Menu(self.axis_menu.menu, tearoff=0)
                for axis_option in self.axis_options:
                    col_menu.add_radiobutton(label=axis_option, variable=var, value=axis_option,
                                           command=lambda c=col, f=file_path, v=var: self.set_column_axis(f, c, v.get()))
                self.axis_menu.menu.add_cascade(label=menu_label, menu=col_menu)
                self.column_axis[file_path][col] = var # Store the Tk variable for later use
        self.axis_menu.pack()

    def set_column_axis(self, file_path, column, axis):
        self.column_axis[file_path][column] = axis
        print(self.column_axis)

    def add_new_axis(self):
        new_axis_number = len(self.axis_options) + 1
        self.axis_options.append(str(new_axis_number))
        self.update_axis_menu()

    def plot_data(self):
        if not self.data_dict:
            messagebox.showwarning("Warning", "No data loaded.")
            return

        selected_indices = self.column_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No columns selected.")
            return

        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        try:
            if self.plot_mode == "time_series":
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)

            self.figure.clear()  # Clear previous plot
            axes = {}  # Store axes objects
            current_axis = "shared"
            axes[current_axis] = self.figure.add_subplot(111)  # Create initial shared axis
            
            for index in selected_indices:
                selected_item = self.column_listbox.get(index)
                file_name, column_name = selected_item.split(": ")
                file_path = [path for path in self.data_dict if file_name in path][0] # Find file path
                data = self.data_dict[file_path]
                if self.plot_mode == "time_series":