import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CSVVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Visualizer")
        self.files = {}
        self.fig = Figure(figsize=(6, 4))
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

        self.axis_selector = ttk.Combobox(self.frame, values=[1, 2, 3, 4], state="readonly")
        self.axis_selector.pack()

        self.update_axis_button = tk.Button(self.frame, text="Update Axis", command=self.update_axis)
        self.update_axis_button.pack()

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
        selected_item = self.metadata_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Select a column to update axis!")
            return
        
        new_axis = self.axis_selector.get()
        if not new_axis:
            messagebox.showwarning("Warning", "Select an axis!")
            return
        
        col_name = self.metadata_tree.item(selected_item, "values")[0]
        self.metadata.loc[self.metadata["Column"] == col_name, "Axis"] = int(new_axis)
        self.metadata_tree.item(selected_item, values=(col_name, new_axis))
        self.plot_data()

    def plot_data(self):
      if self.metadata.empty:
            messagebox.showwarning("Warning", "No columns added to the plot!")
            return
      
      self.ax.clear()
      axes_dict = {}
      colormaps = {1: cm.viridis, 2: cm.plasma, 3: cm.cividis, 4: cm.magma}  # Define colormaps for each axis
      axis_limits = {}
      axis_colors = {}
      
      # Determine min/max values for each axis
      for _, row in self.metadata.iterrows():
            file_path, column, axis = row
            df = self.files[file_path]
            min_val, max_val = df[column].min(), df[column].max()
            
            if axis not in axis_limits:
                  axis_limits[axis] = [min_val, max_val]
            else:
                  axis_limits[axis][0] = min(axis_limits[axis][0], min_val)
                  axis_limits[axis][1] = max(axis_limits[axis][1], max_val)
      
      for i, (_, row) in enumerate(self.metadata.iterrows()):
            file_path, column, axis = row
            df = self.files[file_path]
            
            if axis not in axes_dict:
                  if len(axes_dict) == 0:
                        axes_dict[axis] = self.ax
                  else:
                        axes_dict[axis] = self.ax.twinx()
                        axes_dict[axis].spines['right'].set_position(('outward', i * 15))  # Slight offset
                  
                  cmap = colormaps.get(axis, cm.viridis)  # Get colormap, default to viridis
                  axis_colors[axis] = [cmap(j / (len(self.metadata) + 1)) for j in range(1, len(self.metadata) + 1)]
                  axes_dict[axis].set_ylim(axis_limits[axis])  # Set y limits based on min/max values
            
            ax = axes_dict[axis]
            color = axis_colors[axis].pop(0)
            
            ax.plot(df["Time"], df[column], alpha=0.6, label=f"{column} (Axis {axis})", color=color)
            ax.scatter(df["Time"], df[column], alpha=0.8, s=10, color=color)  # Small markers for visibility
            
            # Set y-axis label and color it accordingly
            ax.set_ylabel(f"Axis {axis}", color=color)
            ax.tick_params(axis='y', colors=color)
            ax.spines['right'].set_color(color)
      
      # Adjust x-ticks to prevent overcrowding
      xticks = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], num=6)
      self.ax.set_xticks(xticks)
      self.ax.tick_params(axis='x', labelrotation=45)
      
      self.ax.legend()
      self.canvas.draw()

      

    def clear_plots(self):
      if hasattr(self, 'fig') and hasattr(self, 'ax'):  # Ensure fig and ax exist
            self.ax.clear()  # Clear main axis
            
            # Remove twin axes if they exist
            for ax in self.ax.figure.axes[1:]:  # Skip the first main axis
                  ax.remove()
            
            # Reset labels, title, grid, and legend
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
            self.ax.set_title("")
            self.ax.grid(False)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Clear metadata
            self.metadata = pd.DataFrame(columns=["File", "Column", "Axis"])
            self.metadata_tree.delete(*self.metadata_tree.get_children())

            # Redraw the canvas
            self.canvas.draw()
      else:
            print("Figure or Axis not initialized!")  # Debugging message


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVVisualizer(root)
    root.mainloop()
