# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:32:25 2024

@author: 21958742
"""


import sys
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
from datetime import datetime
from datetime import timedelta
import Analysis_functions as mage_utils
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation

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
        
        self.geometry('1200x800')

        self.frames = {}

        for F in (StartPage, FileViewer, DataPlotter, TransientSignals):

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
        image = Image.open(r".//assets//MAGE_logo.png")
        img = image.resize((250,250))
        img2 = ImageTk.PhotoImage(img)
        
        label = tk.Label(self, text="MAGE Data Viewer", image=img2)
        label.image = img2
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Choose Files",
                            command=lambda: controller.show_frame(FileViewer))
        button.pack()

        button2 = tk.Button(self, text="Recent Signals",
                            command=lambda: controller.show_frame(TransientSignals))
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

        button2 = tk.Button(self, text="View PSDs",
                            command=lambda: controller.show_frame(DataPlotter))
        button2.pack()
        
        button3 = tk.Button(self, text="Recent Signals",
                            command=lambda: controller.show_frame(TransientSignals))
        button3.pack()
        
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
        selected_item = self.tree.selection()[0]

        if selected_item:
            # Get the text of the selected item (file or folder)
            item_text = self.tree.item(selected_item)['text']
            print(f"Selected: {item_text}")
            
            # Get the parent item to construct the full path
            parent_item = self.tree.parent(selected_item)
            
            # Get the full path of the selected item
            full_path = self.get_full_path(selected_item, parent_item)

            print('Full path in on_select: ' + str(full_path))
            filename, file_extension = os.path.splitext(full_path)
            # Check if the selected item is a file or a folder
            if os.path.isdir(full_path):
                # If it's a directory, open it and list its contents
                self.open_folder(full_path)
            elif file_extension=='.hdf5':
                
                # If it's a file, perform operations on the file (e.g., open or process it)
                self.open_file( r"\\".join(os.getcwd().strip("\\").split('\\')[:-1]) + '\\' + filename + '.hdf5')
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
                NFFT=2**10
                fns = np.zeros((Ninputs, Nchannels, NFFT))
                psd = np.zeros_like(fns)
                for ch in range(Nchannels):
                    dataI_AI0 = file['AI 0/CH ' + str(ch+1)+ '-I/Data'][:]*3.09758E-5**2
                    dataI_AI1 = file['AI 1/CH ' + str(ch+1)+ '-I/Data'][:]*3.09758E-5**2
                    Fs = file['AI 0'].attrs['Fs']
                    fns[0, ch, :], psd[0, ch, :] = welch(dataI_AI0, fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')
                    fns[1, ch, :], psd[1, ch, :] = welch(dataI_AI1, fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')

                # Prepare data for plotting
                # Here we are assuming 'ai_data' and 'ch_data' are numpy arrays
                # You can perform any required processing on this data
                
                # Pass the data to DataPlotter for visualization
                self.controller.frames[DataPlotter].plot_data(fns, psd)
                self.controller.show_frame(DataPlotter)
                
            # Switch to the DataPlotter frame to display the plot
            # self.controller.show_frame(DataPlotter)
        except Exception as e:
            print(f"Error opening file: {e}")
     
import matplotlib.pyplot as plt  # Ensure you import plt

class DataPlotter(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.pack_propagate(False)  # Prevent resizing of the frame based on its contents
        
        label = tk.Label(self, text="Data Viewer")
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()

        back_button = tk.Button(self, text="Back to File Viewer", command=lambda: controller.show_frame(FileViewer))
        back_button.pack()
        
        # Create a Canvas widget that will hold the plot and allow scrolling
        self.scrollable_canvas = tk.Canvas(self)
        
        # Add a vertical scrollbar for the canvas
        self.v_scrollbar = tk.Scrollbar(self, orient="vertical", command=self.scrollable_canvas.yview)
        self.h_scrollbar = tk.Scrollbar(self, orient="horizontal", command=self.scrollable_canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrollable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scrollbar.pack(side=tk.RIGHT, fill="y")
        
        # Configure the scrollbar to work with the canvas
        self.scrollable_canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.scrollable_canvas.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Create a Frame inside the Canvas to hold the plot (this allows for scrolling)
        self.canvas_frame = tk.Frame(self.scrollable_canvas)
        self.scrollable_canvas.create_window((0, 0), window=self.canvas_frame, anchor="nw")
        self.canvas_frame.bind("<Configure>",lambda e: self.scrollable_canvas.configure(scrollregion=self.scrollable_canvas.bbox("all")))



    def plot_data(self, fns, psd):
        """Create 16 subplots in a horizontal scrollable layout"""
        
        if fns.shape != psd.shape or fns.shape[1] != 16 or psd.shape[1] != 16:
            print(f"Error: fns and psd should have the shape (16, 8192), but got {fns.shape} and {psd.shape}.")
            return
        
        n_rows = 4
        n_cols = 8
        self.fig, self.axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(24, 8))
        self.fig.tight_layout(h_pad=2, w_pad=1)

        # Plot each channel's PSD on a separate subplot
        for r in range(n_rows):
            for c in range(n_cols):
                ax = self.axs[r,c]
                ax.plot(psd[r//2,r//2*n_cols + c])
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_title(f"AI {r//2} ch: {n_rows*r + c}", fontsize=11)  
                
        
        
        self.canvas = FigureCanvasTkAgg(self.fig,  master=self.canvas_frame)
        self.canvas.draw()
        plt.close()
        self.canvas_widget = self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            

    def update_contents(self, dirs):
        # Clear the current listbox
        self.content_listbox.delete(0, tk.END)
        # Insert new items
        for item in dirs:
            self.content_listbox.insert(tk.END, item)
            
## Calibrtaion Parameters
def read_two_column_data(file_path):
    data = np.loadtxt(file_path)
    column_1 = data[:, 0]  # First column
    column_2 = data[:, 1]  # Second column
    return np.array([column_1, column_2])
calibration_path = r'C:\Users\00103619\MAGE\MAGE4\calibration'
Vphi = np.transpose(np.genfromtxt(calibration_path + '/Vphi-run7.csv', delimiter=',')/1e6)
Rlambda = read_two_column_data(calibration_path + '/Rs_new.txt')
feffective_mass = open(calibration_path + '/Meff.txt')
mode_distributions = np.genfromtxt(feffective_mass, delimiter=',', skip_header=1)
meff = mode_distributions[:,1]
xi = mode_distributions[:,2]
Lin = 400e-9    # squid calibration parameters
Min = np.array([1 / 0.49 / 1e-6, 1 / 0.517 / 1e-6])
Gamma_max = 20.0 #Maximum accepted linewidth in Hz
error_max = 0.5 #Maximum accepted standard fit error in Hz

class TransientSignals(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Optimal Filter Search for Transients")
        label.pack(pady=10,padx=10)  
        
        button1 = tk.Button(self, text="Choose Folder",
                            command=lambda: controller.show_frame(FileViewer))
        button1.pack()
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        
        # Create the canvas to display the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # Place the canvas in the tkinter window
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.filepath = self.read_most_recent_file()
        self.events = self.find_signals(self.filepath)
        # Function to read the most recent file from the folder
        def read_most_recent_file(self):
            # Get all files in the directory
            files = [f for f in os.listdir(os.getcwd()) if f.endswith('.hdf5')]
            if not files:
                return None
            if len(files) < 1:
                print("Error: No valid files found, reutrn to file browser")
            
            # Get the most recently modified file
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(os.getcwd(), f)))
            
            # Read the data (assuming space/tab-separated text files, or CSV)
            filepath = os.path.join(os.getcwd(), latest_file)
            return filepath
        
        # Computes optimum filter and returns large events
        def find_signals(self, filepath):
            filepath = read_most_recent_file()
            f = h5py.File(filepath, 'r')
            Ninputs = len(f.keys())
            Nchannels = len(f['AI 0'].keys())//2
            Nsample = len(f['AI 0/CH 1-I/Data'][:])
            Fs = f['AI 0'].attrs['Fs']
            dt = 1/Fs
            t_start_string = str(f['AI 0'].attrs['date/time string'])
            t_start = datetime.strptime(t_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
            kb = 1.380649e-23
            NFFT = 2**13
            event_catalogue={}
            large_threshold = 7
            span = 500
            event_data = []
            for AI in range(Ninputs):
                for channel in range(Nchannels):
                    f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
                    
                    dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*9.86e-10
                    dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*9.86e-10
                    Rdat = np.sqrt(dataI**2+dataQ**2)
                    
                    fn, SdataI = welch(dataI, fs=Fs, nperseg = 2*NFFT-1, scaling = 'spectrum')
                    #fn, SdataQ = welch(dataQ, fs=Fs, nperseg = 2*NFFT-1, scaling = 'spectrum')
                    
                    f_res, Gamma, integral, Q, f_res_err, Gamma_err, Q_err, height = mage_utils.lorentzian_fit_thermalpeak_bis(
                        SdataI, fn, f_demod, AI, channel, Plot=False, span=300, noise_ret = False
                        )
                    tau = Q/(np.pi*f_demod)
                    Nfilter=int(Fs*5*tau)
                    
                    if any(Gamma > Gamma_max) or any(Gamma_err > error_max): # Skip channels for which a mode within parameters could not be found
                      continue
                  
                    tn = np.linspace(0,Nsample*dt,Nsample)
                    t_sig = dt*np.linspace(0, Nfilter, Nfilter)

                    G = (Vphi[AI, channel]*2000)*Min[AI]
                    kappa = np.sqrt(f_demod*2*np.pi*meff[channel]/(Q) * Rlambda[AI,channel])
                    template = np.exp(-t_sig/(tau)) # template construction                    
                    h = np.fft.ifft(np.fft.fft(Rdat[1:]/G)/(kappa*f_demod*2*np.pi)).real
                    T = np.mean(0.5*(f_demod*2*np.pi)**2*meff[channel]*h**2/kb)
                    SNR, dat_filt = mage_utils.optimal_filter(h, template, Fs, NFFT)
                    threshold = 0.5 ## effective noise temperature
                    peaks = find_peaks(SNR, height = threshold, distance = int(tau*Fs), width = [100, 5e6], rel_height=1.0)

                    event_day_string = datetime.strftime(t_start, "%d-%m-%y")
                    
                    ## data quality cuts
                    if len(peaks[0])>0:
                        diverge_template1 = np.exp(-t_sig/(tau/10.0))
                        diverge_template2 = np.exp(-t_sig/(tau*10.0))
                        transient_SNR1, junk = mage_utils.optimal_filter(h, diverge_template1, Fs, NFFT)           
                        transient_SNR2, junk = mage_utils.optimal_filter(h, diverge_template2, Fs, NFFT)
                    for event_i in peaks[0]:
                        #print('%1.2f'%(SNR[event_i]) + ', %1.2f' % (transient_SNR1[event_i]) + ', %1.2f' % (transient_SNR2[event_i]))
                        if ((SNR**2)[event_i] < (transient_SNR1**2)[event_i] or (SNR**2)[event_i] < (transient_SNR2**2)[event_i]):
                            continue
                        if SNR[event_i] < large_threshold:
                            continue
                        event_time = t_start + timedelta(seconds = event_i*dt)
                        event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-AI" + str(AI) + "-ch" + str(channel+1) + "-SNR %1.2f" % (SNR[event_i])
                        event_data.append(np.array([SNR[event_i-span:event_i+span], tn[event_i-span:event_time+span] + t_start.timestamp()]))
            return event_data
                        
        def update_plot(self):
            ydata = self.events
            
if __name__ == "__main__":
    app = mainFrame()
    #ani = animation.FuncAnimation(f,app.frames[DataPlotter].animate, interval=1000)
    app.mainloop()
            
