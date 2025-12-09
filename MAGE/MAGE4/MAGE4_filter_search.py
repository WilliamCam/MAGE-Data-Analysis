from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from os import listdir
from scipy import signal
from scipy.signal import welch, csd
from scipy.signal import find_peaks
import lmfit
from scipy.fft import fft, fftfreq
import scipy.constants
from lmfit.models import LorentzianModel, ConstantModel
import sys
import os
import numpy as np
import re
import h5py
library_path = r'C:\Users\21958742\GitHub\MAGE-Data-Analysis\MAGE\MAGE4/'  # Adjust this to the actual absolute path
#library_path = '/home/leo_maria/Desktop/UWA/MAGE/libraries/'

sys.path.append(library_path)
import Analysis_functions

#file by file data analysis
harddisk_fold = r"C:\Users\21958742\MAGE"
## Transient filter search for MAGE data stream. First test designed for MAGE0 Data

#load data file
folder = r"C:\Users\21958742\MAGE"
#folder='/home/leo_maria/Desktop/UWA/MAGE'
exp_name = "MAGE4"
run_name = "run3"
identifier = 'run3-'

files = listdir(harddisk_fold + '/' + exp_name + '/' + run_name)
numfile = np.zeros(len(files))
# Regular expression to match an integer after a hyphen
for i,file in enumerate(files):
    match = re.search(r'-(\d+)', file)
    if match:
        number = int(match.group(1))  # Convert the matched string to an integer
        numfile[i] = number

    else:
        print("No integer found after a hyphen.")

Nfiles = len(files)

#meta data from first file
f = h5py.File(harddisk_fold + '/' + exp_name + '/' + run_name + '/' + files[0], 'r')
f1 = h5py.File(harddisk_fold + '/' + exp_name + '/' + run_name + '/' + files[1], 'r')

#load calibration data
def read_two_column_data(file_path):
    data = np.loadtxt(file_path)
    column_1 = data[:, 0]  # First column
    column_2 = data[:, 1]  # Second column
    return np.array([column_1, column_2])

Vphi = read_two_column_data(folder + '/' + exp_name + '/calibration/Vphi_squids_quartz_order.txt')
Rlambda = read_two_column_data(folder + '/' + exp_name + '/calibration/Rs_new.txt')
feffective_mass = open(folder + '/' + exp_name + '/calibration/Meff.txt')
mode_distributions = np.genfromtxt(feffective_mass, delimiter=',', skip_header=1)
meff = mode_distributions[:,1]
xi = mode_distributions[:,2]
Lin = 400e-9    # squid cali parameters
Min = np.array([1 / 0.49 / 1e-6, 1 / 0.517 / 1e-6])

Ninputs = len(f.keys())
Nchannels = len(f['AI 0'].keys())//2
Nsample = len(f1['AI 0/CH 1-I/Data'][:])
Fs = f['AI 0'].attrs['Fs']
dt = 1/Fs
data_array = np.zeros((Ninputs, Nchannels, 2, Nsample))
strain = np.zeros((Nsample))
t_start_string = str(f['AI 0'].attrs['date/time string'])
t_start = datetime.strptime(t_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')

kb = 1.380649e-23
event_catalogue={}
candidate_events = []
#create output folder to store analysis results
output_path = folder + '/' + exp_name + '/Analysis/' + run_name + '-strain'
if os.path.exists(output_path) == False:
    os.makedirs(output_path)

noise_input = False # replaces data with random noise

#filter function
def optimal_filter(data, template, Fs, NFFT):
    fft = np.fft.fft(data) # fourier transformed data
    zero_pad = np.zeros(data.size - template.size) # zero pad template to match data size
    template_pad = np.append(template, zero_pad)
    fft_template = np.fft.fft(template_pad) # fourier transformed padded template
    power_dat, freq_PSD = plt.psd(data, Fs=Fs, NFFT = NFFT, visible = True)
    freq_dat = np.fft.fftfreq(data.size)*Fs #fourier frequencies corresponding to data partition
    power_spec = np.interp(freq_dat, freq_PSD, power_dat)
    
    val_cal = np.max(template)
    OF = np.fft.ifft(fft_template*fft_template.conjugate()/power_spec).real
    K = val_cal/np.amax(OF)
    
    
    df = np.abs(freq_dat[1] - freq_dat[2])
    opt_filter = K * fft * fft_template.conjugate() / power_spec #optimal filter
    dat_filt = np.fft.ifft(opt_filter) #revert to time domain for filter output
    
    sigmasq = 2*(K**2 * fft_template * fft_template.conjugate() / power_spec).sum() * df 
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = np.abs(2*dat_filt) / (sigma)
    return SNR, dat_filt

# create numpy array with all data
start_file=1
for file in range(300, Nfiles-1): #Current version of MAGE.vi gives false data in first file
    f = h5py.File(harddisk_fold + '/' + exp_name + '/' + run_name + '/' + identifier + str(int(file)) + '.hdf5', 'r')
    file_start = t_start + timedelta(seconds = file*Nsample*dt)
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*9.595e-10
            dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*9.595e-10
            
            data_array[AI, channel, 0, :] = dataI
            data_array[AI, channel, 1, :] = dataQ

    #Determine single sided power spectrum for each stream
    NFFT = 2**12 # for NFFT < Nsample power spectrum will be averaged
    
    Sx = np.zeros((Ninputs, Nchannels, NFFT))
    Sy = np.zeros((Ninputs, Nchannels, NFFT))
    
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            fn, SdataI = welch(data_array[AI, channel, 0, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')
            fn, SdataQ = welch(data_array[AI, channel, 1, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')
            
            Sx[AI, channel, :] = SdataI
            Sy[AI, channel, :] = SdataQ
            
    
    #Fit lorentzian to each stream to determine Q and fpeak
    from lmfit.models import LorentzianModel, ConstantModel
    
    Q_array = np.zeros((Ninputs, Nchannels, 2))
    Gamma_array = np.zeros(Q_array.shape)
    height_array = np.zeros(Q_array.shape)
    fcenter_array = np.zeros(Q_array.shape)
    error_array = np.zeros(Q_array.shape)
    Gamma_max = 20.0 #Maximum accepted linewidth in Hz
    error_max = 0.5 #Maximum accepted standard fit error in Hz
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            Sxdat = Sx[AI,channel,:]
            peak_ii = np.where(Sxdat==np.max(Sxdat))[0][0]
            ii_span = 2**10
            
            ss = peak_ii-ii_span//2
            tt = peak_ii+ii_span//2
            if ss < 1:
                ss = 0
                        
            Sx_trim = Sx[AI, channel, ss:tt]
            Sy_trim = Sy[AI, channel, ss:tt]
            fn_trim = fn[ss:tt]
            
    
            peak_model = LorentzianModel()
            noise_model = ConstantModel()
    
            pars = peak_model.guess(Sx_trim, x=fn_trim)
            pars += noise_model.guess(Sx_trim, x=fn_trim)
            pars2 = peak_model.guess(Sx_trim, x=fn_trim)
            pars2 += noise_model.guess(Sx_trim, x=fn_trim)
            model = peak_model + noise_model
    
            out = model.fit(Sx_trim, pars, x=fn_trim)
            out2 = model.fit(Sy_trim, pars2, x=fn_trim)
            Gamma1, Gamma2 = out.params["sigma"].value, out2.params["sigma"].value
            fcenter1, fcenter2 = out.params["center"].value, out2.params["center"].value
            h1, h2 = out.params["height"].value + out.params["c"].value, out2.params["height"].value + out2.params["c"].value
            f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
    
            Q1, Q2 = (fcenter1+f_demod)/(2*Gamma1), (fcenter2+f_demod)/(2*Gamma2)
            if out.params["sigma"].stderr == None:
                error1 = 0.99
            else:
                error1 =  out.params["sigma"].stderr/out.params["sigma"].value
            if out2.params["sigma"].stderr == None:
                error2 = 0.99
            else:
                error2 =  out2.params["sigma"].stderr/out2.params["sigma"].value
            Q_array[AI, channel, :] = [Q1,Q2]
            Gamma_array[AI, channel, :] = [Gamma1,Gamma2]
            height_array[AI, channel, :] = [h1*(1+(fcenter1/151.8)),h2*(1+(fcenter2/151.8))]
            fcenter_array[AI, channel, :] = [fcenter1, fcenter2]
            error_array[AI, channel, :] = [error1, error2]
            if Gamma1 > Gamma_max or Gamma2 > Gamma_max or error1 > error_max or error2 > error_max: #Ignore bad fits
                print("Input AI " + str(AI) + ", Channel " + str(channel+1) + ":WARNING: Bad mode detected, channel will be ignored")

            plt.ion()
            fig = plt.figure("IMPA DOWNLOAD")
            plt.axis('tight')
            plt.pause(0.05)
            plt.draw()
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(fn_trim, Sx_trim, 'o', markersize=0.2)
            ax.set_title("Input AI " + str(AI) + ", Channel " + str(channel+1))
            #plt.plot(fn_n, out.init_fit, '--', label='initial fit')
            ax.plot(fn_trim, out.best_fit, '-', label='best fit I')
            ax.plot(fn_trim, out2.best_fit, '-', label='best fit Q')
            ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.legend()
    
    
    #Filtering data
    Nfilter=NFFT
    tn = np.linspace(0,Nsample*dt,Nsample)
    t_sig = dt*np.linspace(0, Nfilter, Nfilter)

    event_catalogue_perfile = {}
    for AI in range(Ninputs):
        for channel in range(Nchannels):

            f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
            tau1, tau2 = Q_array[AI, channel]/(np.pi*f_demod)
            fpeak_sig = fcenter_array[AI, channel, 0]
            
            if any(Gamma_array[AI, channel, :] > Gamma_max) or any(error_array[AI, channel, :] > error_max): # Skip channels for which a mode within parameters could not be found
              continue
          
            dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*9.86e-10
            dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*9.86e-10
            
            if noise_input:
                dataI_noise = np.random.normal(np.mean(dataI), np.std(dataI), size = len(dataI))
                dataQ_noise = np.random.normal(np.mean(dataQ), np.std(dataQ), size = len(dataQ))
                Rdat = np.sqrt(dataI_noise**2+dataQ_noise**2)
            else:    
                Rdat = np.sqrt(dataI**2+dataQ**2)
            f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
            
            Nfilter=int(Fs*5*tau1)
            tn = np.linspace(0,Nsample*dt,Nsample)
            t_sig = dt*np.linspace(0, Nfilter, Nfilter)
            kappa = np.sqrt(f_demod*2*np.pi*meff[channel]/(np.mean(Q_array[AI,channel]) * Rlambda[AI,channel]))
            template = np.exp(-t_sig/(tau1)) # template construction
            h = Analysis_functions.R_to_strain(Rdat, Fs, f_demod, Vphi[AI,channel], Min[AI], kappa)
            SNR, dat_filt = optimal_filter(h, template, Fs, NFFT)
            '''
            G = squid_model[AI][0]/np.sqrt(1+(f_demod/squid_model[AI][1])**2) #squid gain

            
            kappa = np.sqrt(f_demod*2*np.pi*meff[channel]/(np.mean(Q_array[AI, channel])*resistances[channel][AI]))
            
            template = np.exp(-t_sig/(tau1)) # template construction
            
            h = np.fft.ifft(np.fft.fft(Rdat[1:]/G)/(kappa*f_demod*2*np.pi)).real
            
            Zl = 1j*2*np.pi*f_demod*400e-9 #circuit input impedance
            Zc = np.abs(1/(1j*2*np.pi*f_demod*4e-12))
            ZlZc = (Zl+Zc)/(Zl*Zc)
            R = resistances[channel][AI]
            Zi = ZlZc + R
            VtoT = 1/G*np.abs(Zi)/np.sqrt(4*kb*R)
            T = np.mean(height_array[AI, channel, :])*VtoT**2
            SNR, dat_filt = optimal_filter(h, template, Fs, NFFT)
            '''
            threshold = 0.5 ## effective noise temperature
            peaks = find_peaks(SNR, height = threshold, distance = int(tau1*Fs), width = [100, 5e6], rel_height=1.0)
            
            # file_start_date = t_start + timedelta(seconds = file*Nsample*dt)
            event_day_string = datetime.strftime(file_start, "%d-%m-%y")
            
            ## data quality cuts
            if len(peaks[0])>0:
                diverge_template1 = np.exp(-t_sig/(tau1/10.0))
                diverge_template2 = np.exp(-t_sig/(tau1*10.0))
                transient_SNR1, junk = optimal_filter(h, diverge_template1, Fs, NFFT)           
                transient_SNR2, junk = optimal_filter(h, diverge_template2, Fs, NFFT)
                plt.ion()
            # fig = plt.figure("Filtered Output")
            # plt.pause(0.05)
            # plt.draw()
            # fig.clf()
            # ax = fig.add_subplot(111)
            # ax.set_title("File " + str(file) + " Input AI " + str(AI) + ", Channel " + str(channel+1))
            # ax.plot(tn, Tn/np.mean(Tn), label = 'Normalised Mangitude')
            # ax.plot(tn, SNR, label = 'SNR')
            # ax.legend()
            #ax.plot(peaks[0]*dt,peaks[1]['peak_heights'], linestyle = ' ', marker = 'x', color = 'black')
           
            for event_i in peaks[0]:
                #print('%1.2f'%(SNR[event_i]) + ', %1.2f' % (transient_SNR1[event_i]) + ', %1.2f' % (transient_SNR2[event_i]))
                if ((SNR**2)[event_i] < (transient_SNR1**2)[event_i] or (SNR**2)[event_i] < (transient_SNR2**2)[event_i]):
                    print("Transiently divergent feature detected of SNR %1.2f" % (SNR[event_i]) + ", performing quality cut")
                    continue
                print("Large event detected" + '\n')
                print("File " + str(file) + " Input AI " + str(AI) + ", Channel " + str(channel+1))
                event_time = file_start + timedelta(seconds = event_i*dt)
                event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-AI" + str(AI) + "-ch" + str(channel+1) + "-SNR %1.2f" % (SNR[event_i])
                if event_name not in event_catalogue: 
                    event_catalogue[event_name] = {'time' : event_time, 'SNR' : SNR[event_i], 'input AI' : AI, 'channel' : channel+1, 'frequency' : f_demod, 'amplitude' : dat_filt[event_i], 'file N' : file, 'index' : event_i}
                    print('Event ' + event_name + ' Saved')
                if event_name not in event_catalogue_perfile:
                    event_catalogue_perfile[event_name] = {'time' : event_time, 'SNR' : SNR[event_i], 'input AI' : AI, 'channel' : channel+1, 'frequency' : f_demod, 'amplitude' : dat_filt[event_i], 'file N' : file, 'index' : event_i}
    # Coincident modes on one file
    print("Looking for Coincident events...")
    times1 =  [event_catalogue_perfile[event]['time'].timestamp() for event in event_catalogue_perfile if (event_catalogue_perfile[event]['input AI'] == 1)]
    times0 =  [event_catalogue_perfile[event]['time'].timestamp() for event in event_catalogue_perfile if (event_catalogue_perfile[event]['input AI'] == 0)]
    
    coincident_t = []
    for time0 in times0:
        for time1 in times1:
            if np.abs(time0-time1) < 0.05:
                #print("Coincident Event at " + str(time0))
                coincident_t.append(time0)
                break
    for ii in range(len(coincident_t)):
        co_event_nn = ii
        co_event = [(event, event_catalogue_perfile[event]) for event in event_catalogue_perfile if event_catalogue_perfile[event]['time'] == datetime.fromtimestamp(coincident_t[co_event_nn])]
        co_event0 = [event for event in co_event if event[1]['input AI']==0]
        co_event1 = [event for event in co_event if event[1]['input AI']==1]
        
        for event0 in co_event0:
            for event1 in co_event1:
                if event0[1]['channel'] == event1[1]['channel']:
                    candidate_events.append([event0,event1])                    
    print("Candidate events :" + str(len(candidate_events)))
import pickle
with open(output_path + '/event_catalogue-strain-SecondHalf.pkl', 'wb') as f:
    pickle.dump(event_catalogue, f)      
with open(output_path + '/co_event_strain-SecondHalf.pkl', 'wb') as f:
    pickle.dump(candidate_events, f)         
    ## Plot filtered results / mode temperatures --> wont be accurate for MAGE0 Data
    
    ## Transient search for Impulse events some decay shape (N consecutive samples with T>?)
    
    ## 
