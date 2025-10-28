import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class CSVVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Visualizer")
        self.files = {}
        self.metadata = pd.DataFrame(columns=["File", "Column", "Axis"])
        
        self.setup_ui()

    def setup_ui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.load_button = tk.Button(self.frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack()

        self.file_column_frame = tk.Frame(self.frame)
        self.file_column_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(self.file_column_frame, exportselection=False)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.update_column_selection)

        self.column_listbox = tk.Listbox(self.file_column_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.column_listbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.add_button = tk.Button(self.frame, text="Add to Plot", command=self.add_to_plot)
        self.add_button.pack()

        self.metadata_tree = ttk.Treeview(self.frame, columns=("Column", "Axis"), show='headings')
        self.metadata_tree.heading("Column", text="Column")
        self.metadata_tree.heading("Axis", text="Axis")
        self.metadata_tree.pack(fill=tk.BOTH, expand=True)

        self.remove_button = tk.Button(self.frame, text="Remove Selected", command=self.remove_selected)
        self.remove_button.pack()

        self.plot_button = tk.Button(self.frame, text="Plot Data", command=self.plot_data)
        self.plot_button.pack()

        self.clear_button = tk.Button(self.frame, text="Clear Plots", command=self.clear_plots)
        self.clear_button.pack()

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            self.files[file_path] = df
            self.file_listbox.insert(tk.END, file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def update_column_selection(self, event):
        selected_index = self.file_listbox.curselection()
        if not selected_index:
            return
        
        file_path = self.file_listbox.get(selected_index[0])
        df = self.files[file_path]
        
        self.column_listbox.delete(0, tk.END)
        for col in df.columns:
            if col != "Time":
                self.column_listbox.insert(tk.END, col)

    def add_to_plot(self):
        selected_index = self.file_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "No file selected!")
            return

        file_path = self.file_listbox.get(selected_index[0])
        selected_columns = [self.column_listbox.get(i) for i in self.column_listbox.curselection()]
        
        if not selected_columns:
            messagebox.showwarning("Warning", "Select at least one column!")
            return

        for col in selected_columns:
            self.metadata = pd.concat([self.metadata, pd.DataFrame([[file_path, col, 1]],
                                                                    columns=["File", "Column", "Axis"])],
                                      ignore_index=True)
            self.metadata_tree.insert("", tk.END, values=(col, 1))

    def remove_selected(self):
        selected_item = self.metadata_tree.selection()
        if not selected_item:
            return
        
        col_name = self.metadata_tree.item(selected_item, "values")[0]
        self.metadata = self.metadata[self.metadata["Column"] != col_name]
        self.metadata_tree.delete(selected_item)
        self.plot_data()

    def update_axis(self):
        messagebox.showwarning("Warning", "Select an axis!")

    def plot_data(self):
        if self.metadata.empty:
            messagebox.showwarning("Warning", "No columns added to the plot!")
            return
        
        self.ax.clear()
        axes_dict = {}
        
        for _, row in self.metadata.iterrows():
            file_path, column, axis = row
            df = self.files[file_path]
            
            if axis not in axes_dict:
                if len(axes_dict) == 0:
                    axes_dict[axis] = self.ax
                else:
                    axes_dict[axis] = self.ax.twinx()
                
            axes_dict[axis].plot(df["Time"], df[column], label=f"{column} (Axis {axis})")
            axes_dict[axis].set_ylabel(f"Axis {axis}")
        
        self.ax.legend()
        self.canvas.draw()

    def clear_plots(self):
        self.ax.clear()
        self.metadata = pd.DataFrame(columns=["File", "Column", "Axis"])
        self.metadata_tree.delete(*self.metadata_tree.get_children())
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVVisualizer(root)
    root.mainloop()
