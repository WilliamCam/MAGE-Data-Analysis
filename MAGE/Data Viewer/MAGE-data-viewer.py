# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:32:25 2024

@author: 21958742
"""

import tkinter as tk
import utilities
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
import h5py
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation

f = Figure(figsize=(5,4), dpi=100)
ax = f.add_subplot(111)


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
        image = Image.open(r"assets/MAGE_logo.png")
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
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.populate_tree(folder_path)
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

    def on_select(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            item_text = self.tree.item(selected_item)['text']
            print(f"Selected: {item_text}")
            parent_item = self.tree.parent(selected_item)
            # Get the full path of the selected file
            full_path = "../" + self.get_full_path(selected_item, parent_item)

            print(f"Opening: {full_path}")
            print(os.getcwd())
            self.open_file(full_path)
    
       

    def get_full_path(self, selected_item, parent_item):
        path_parts = []
        while parent_item:
            path_parts.append(self.tree.item(parent_item)['text'])
            parent_item = self.tree.parent(parent_item)
        path_parts.reverse()
        return os.path.join(*path_parts, self.tree.item(selected_item)['text'])

    def open_file(self, full_path):
        try:
            print(full_path)
            dirs = os.listdir(full_path)
            # Show the new frame with the directory contents
            self.controller.frames[DataPlotter].update_contents(dirs) # plot selected data
            if self.controller.frames[DataPlotter].meta_data_found(dirs)==False: # to be added later
                print(full_path + " is yet to be analysed.")
            #self.controller.show_frame(DataPlotter)
        except Exception as e:
            print(f"Error opening folder: {e}")


class DataPlotter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller=controller
        label = tk.Label(self, text="Data Viewer")
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        
        self.content_listbox = tk.Listbox(self)
        self.content_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        back_button = tk.Button(self, text="Back to File Viewer",
                                command=lambda: controller.show_frame(FileViewer))
        back_button.pack()
        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
    
#     def select_data(self):
#         file = self.content_listbox.curselection
#         self.data_ch = utilities.data_channel(self, file, AI, channel)
    
#     def animate(i):
#         f = data_ch.File
#         dataI = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-I/Data'][:]*9.86e-10
#         dataQ = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-Q/Data'][:]*9.86e-10
#         ax.plot(dataI)
#         ax.plot(dataQ)
#         ax.plot()
        
# try:
#     from ctypes import windll

#     windll.shcore.SetProcessDpiAwareness(1)
# finally:
#     if __name__ == "__main__":
#         app = mainFrame()
#         ani = animation.FuncAnimation(f,app.frames[DataPlotter].animate, interval=1000)
#         app.mainloop()
            
