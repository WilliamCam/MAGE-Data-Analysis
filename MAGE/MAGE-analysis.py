# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:32:25 2024

@author: 21958742
"""

import tkinter as tk


root = tk.Tk()


# place a label on the root window
message = tk.Label(root, text="Hello, World!")
message.pack()

root.title("MAGE Data Analysis Viewer")

root.geometry('600x400+50+50')

try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
finally:
    root.mainloop()
