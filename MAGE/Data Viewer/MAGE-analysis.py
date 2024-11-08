# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:32:25 2024

@author: 21958742
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
import h5py


class mainFrame(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.geometry('600x400')

        self.frames = {}

        for F in (StartPage, FileViewer, PageTwo):

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
        image = Image.open("./assets/MAGE_logo.png")
        img = image.resize((250,250))
        img2 = ImageTk.PhotoImage(img)
        
        label = tk.Label(self, text="MAGE Data Viewer", image=img2)
        label.image = img2
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Choose Files",
                            command=lambda: controller.show_frame(FileViewer))
        button.pack()

        button2 = tk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        


class FileViewer(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Select MAGE run folder containing hdf5 files")
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
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
            full_path = self.get_full_path(selected_item, parent_item)

            if os.path.isfile(full_path):
                print(f"Opening: {full_path}")
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
           os.path.isdir(full_path)
        except Exception as e:
            print(f"Error opening folder: {e}")
        else:
            dirs = os.listdir(full_path)
        return dirs


class DataPlotter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!")
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Choose Files",
                            command=lambda: controller.show_frame(FileViewer))
        button2.pack()
        
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
finally:
    if __name__ == "__main__":
        app = mainFrame()
        app.mainloop()
            
