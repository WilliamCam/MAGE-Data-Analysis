# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:32:25 2024

@author: 21958742
"""


import sys
# Specify the absolute path to the 'libraries' directory
library_path = '/home/leo_maria/Desktop/UWA/MAGE/'  # Adjust this to the actual absolute path

# Add the 'libraries' directory to sys.path
sys.path.append(library_path)

# Now you can import utils.py as a module
import utilities
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from scipy.signal import welch
f = Figure(figsize=(5,4), dpi=100)
ax = f.add_subplot(111)

logo_path = '/home/leo_maria/Desktop/UWA/figures/MAGE_logo.png'
initial_dir_path = '/home/leo_maria/Desktop/UWA/MAGE/MAGE4'
rootdirectory_path = '/home/leo_maria/Desktop/UWA/MAGE'

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class mainFrame(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.geometry('600x400')

        self.frames = {}

        for F in (StartPage, FileViewer, DataPlotter):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
        self.title("MAGE Data Viewer")

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        image = Image.open(logo_path)
        img = image.resize((250,250))
        img2 = ImageTk.PhotoImage(img)
        
        label = tk.Label(self, text="MAGE Data Viewer", image=img2)
        label.image = img2
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Choose Files",
                            command=lambda: controller.show_frame(FileViewer))
        button.pack()

        button2 = tk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(DataPlotter))
        button2.pack()
        


class FileViewer(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Select MAGE run folder containing hdf5 files")
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="View Data",
                            command=lambda: controller.show_frame(DataPlotter))
        button2.pack()
        
        # Create a Treeview
        self.tree = ttk.Treeview(self)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Bind the treeview selection
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        

        # Create a button to browse folders
        self.browse_button = tk.Button(self, text="Browse Folder", command=self.browse_folder)
        self.browse_button.pack(pady=10)
        
        
    def browse_folder(self):
        folder_path = filedialog.askdirectory(initialdir=initial_dir_path)
        if folder_path:
            self.populate_tree(folder_path)
            print('tree popuated from : ' +str(folder_path))
            os.chdir(folder_path)

    def populate_tree(self, path):
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert the root node
        root_node = self.tree.insert('', 'end', text=os.path.basename(path), open=True)
        self.insert_subfolders(root_node, path)

    def insert_subfolders(self, parent, path):
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    node = self.tree.insert(parent, 'end', text=item, open=False)
                    self.insert_subfolders(node, full_path)
                else:
                    self.tree.insert(parent, 'end', text=item)
        except PermissionError:
            pass  # Handle the case where access is denied

    import os

    def on_select(self, event):
        # Get the selected item from the treeview
        selected_item = self.tree.selection()

        if selected_item:
            # Get the text of the selected item (file or folder)
            item_text = self.tree.item(selected_item)['text']
            print(f"Selected: {item_text}")
            
            # Get the parent item to construct the full path
            parent_item = self.tree.parent(selected_item)
            
            # Get the full path of the selected item
            full_path = self.get_full_path(selected_item, parent_item)
            print('Full path in on_select: ' + str(full_path))

            # Check if the selected item is a file or a folder
            if os.path.isdir(full_path):
                # If it's a directory, open it and list its contents
                self.open_folder(full_path)
            elif os.path.isfile(full_path):
                # If it's a file, perform operations on the file (e.g., open or process it)
                self.open_file(full_path)
            else:
                print("Selected item is neither a file nor a folder.")

    def open_folder(self, full_path):
        # Open a folder, list its contents, and update the UI
        try:
            dirs = os.listdir(full_path)
            print(f"Contents of folder {full_path}:")
            # Show the contents in the DataPlotter frame
            self.controller.frames[DataPlotter].update_contents(dirs)
            if not self.controller.frames[DataPlotter].meta_data_found(dirs):
                print(full_path + " is yet to be analyzed.")
        except Exception as e:
            print(f"Error opening folder: {e}")

    def open_file(self, full_path):
        # Perform operations on the file, such as opening or processing it
        try:
            print(f"Opening file: {full_path}")
            with h5py.File(full_path, 'r') as file:
                # Extract data from the file (assuming the structure of the file)
                # For example, you might have 'AI' and 'CH' data
                Ninputs=2
                Nchannels=16
                Nsamples = len(file['AI 0/CH 1-I/Data'][:])
                NFFT=2**13
                fns = np.zeros((Ninputs, Nchannels, NFFT ))
                psd = np.zeros_like(fns)
                for ch in range(Nchannels):
                    dataI = file['AI 1/CH ' + str(ch+1)+ '-I/Data'][:]*3.09758E-5**2
                    Fs = file['AI 0'].attrs['Fs']
                    fns[0, ch, :], psd[0, ch, :] = welch(dataI, fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')

                # Prepare data for plotting
                # Here we are assuming 'ai_data' and 'ch_data' are numpy arrays
                # You can perform any required processing on this data
                
                # Pass the data to DataPlotter for visualization
                #self.controller.frames[DataPlotter].plot_psd(fns, psd)
                self.controller.frames[DataPlotter].plot_data(fns, psd)
            # Switch to the DataPlotter frame to display the plot
            self.controller.show_frame(DataPlotter)
        except Exception as e:
            print(f"Error opening file: {e}")
     

    def get_full_path(self, selected_item, parent_item):
        path_parts = []

        # Start with the current selected item
        path_parts.append(self.tree.item(selected_item)['text'])
        
        # Traverse upwards to the root to construct the full path
        parent_item = self.tree.parent(selected_item)
        while parent_item:
            path_parts.append(self.tree.item(parent_item)['text'])
            parent_item = self.tree.parent(parent_item)
        
        # Reverse the path to get it from root to the selected item
        path_parts.reverse()

        # Debugging: print out the collected parts
        print(f"Path parts: {path_parts}")

        # Prepend the absolute root directory before joining
        root_directory = rootdirectory_path  # Adjust this path to your root directory
        full_path = os.path.join(root_directory, *path_parts)

        print(f"Full path: {full_path}")
        
        return full_path


import matplotlib.pyplot as plt  # Ensure you import plt

class DataPlotter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Data Viewer")
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()

        back_button = tk.Button(self, text="Back to File Viewer", command=lambda: controller.show_frame(FileViewer))
        back_button.pack()

        # Create a canvas to hold the scrolling content
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar for the canvas
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a horizontal scrollbar for the canvas
        self.scrollbar_x = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Configure the canvas to use both scrollbars
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        # Create a frame that will hold the plot(s)
        self.plot_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")

    def plot_data(self, fns, psd):
        """Create 16 subplots in a horizontal scrollable layout"""
        
        if fns.shape != psd.shape or fns.shape[1] != 16 or psd.shape[1] != 16:
            print(f"Error: fns and psd should have the shape (16, 8192), but got {fns.shape} and {psd.shape}.")
            return
        
        n_rows = 1
        n_cols = 16
        self.fig, self.axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(120, 8))
        self.fig.tight_layout(pad=4.0)

        # Plot each channel's PSD on a separate subplot
        for c in range(n_cols):
            ax = self.axs[c]
            ax.plot(psd[0,c])
            ax.set_title(f"PSD for channel {c}")        

        # Create a canvas for the Matplotlib figure and pack it into the plot frame
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget.draw()

        # Pack the canvas widget inside the frame to make it scrollable
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Update the scroll region to match the total width of the plots
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        bbox = self.canvas.bbox("all")
        print(f"Canvas bbox: {bbox}")  # Debugging: Check the bounding box of the canvas
    


    def plot_psd(self, fns, psd):
        # Clear the axes and plot the new data
        self.ax.clear()
        self.ax.plot(fns, psd, label='PSD of I quad, AI 0, channel 0', color='blue')

        # Customize plot
        self.ax.set_title("Example PSD")
        self.ax.set_xlabel("Freqs[Hz]")
        self.ax.set_ylabel("PSD[V/rtHz]")
        self.ax.legend()

        # Redraw the canvas with the updated plot
        self.canvas.draw()

    def update_contents(self, dirs):
        # Clear the current listbox
        self.content_listbox.delete(0, tk.END)
        # Insert new items
        for item in dirs:
            self.content_listbox.insert(tk.END, item)
            
    def meta_data_found(self, dirs):
        ret = False
        for item in dirs:
            if item[-3:] == 'pkl':
                ret = True
                break
        return ret
    
    def select_data(self):
        file = self.content_listbox.curselection
        self.data_ch = utilities.data_channel(self, file, AI, channel)
    
    def animate(i):
        f = data_ch.File
        dataI = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-I/Data'][:]*9.86e-10
        dataQ = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-Q/Data'][:]*9.86e-10
        ax.plot(dataI)
        ax.plot(dataQ)
        ax.plot()
        

if __name__ == "__main__":
    app = mainFrame()
    #ani = animation.FuncAnimation(f,app.frames[DataPlotter].animate, interval=1000)
    app.mainloop()
            
