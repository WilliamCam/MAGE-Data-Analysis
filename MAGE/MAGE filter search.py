import h5py
from scipy import signal
from scipy.signal import welch, csd
import matplotlib
import matplotlib.pyplot as plt
import os
from os import listdir
import numpy as np
from scipy import fft
from scipy.signal import find_peaks
import lmfit
matplotlib.use('TkAgg')
from datetime import datetime
from datetime import timedelta
#file by file data analysise

## Transient filter search for MAGE data stream. First test designed for MAGE0 Data

#load data file
folder = r"C:\Users\21958742\MAGE/"
exp_name = "MAGE3"
run_name = "run2"
identifier = 'run2-'

files = listdir(folder + '/' + exp_name + '/' + run_name)
Nfiles = len(files)

#meta data from first file
f = h5py.File(folder + '/' + exp_name + '/' + run_name + '/' + files[0], 'r')
f1 = h5py.File(folder + '/' + exp_name + '/' + run_name + '/' + files[1], 'r')

#load calibration data
fsquid = open(folder + '/' + exp_name + '/calibration/SQUID gain models.txt')
fresistance = open(folder + '/' + exp_name + '/calibration/Rs.txt')
squid_model = np.genfromtxt(fsquid, skip_header=1, delimiter=',')
resistances = np.genfromtxt(fresistance, delimiter=',')

Ninputs = len(f.keys())
Nchannels = len(f['AI 0'].keys())//2
Nsample = len(f1['AI 0/CH 1-I/Data'][:])
Fs = f['AI 0'].attrs['Fs']
dt = 1/Fs
data_array = np.zeros((Ninputs, Nchannels, 2, Nsample))
t_start_string = str(f['AI 0'].attrs['date/time string'])
t_start = datetime.strptime(t_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')

kb = 1.380649e-23
event_catalogue={}
#create output folder to store analysis results
output_path = folder + '/' + exp_name + '/Analysis/' + run_name + '-1 sigma'
if os.path.exists(output_path) == False:
    os.makedirs(output_path)

#filter function
def optimal_filter(data, template, Fs, NFFT):
    fft = np.fft.fft(data) # fourier transformed data
    zero_pad = np.zeros(data.size - template.size) # zero pad template to match data size
    template_pad = np.append(template, zero_pad)
    fft_template = np.fft.fft(template_pad) # fourier transformed padded template
    power_dat, freq_PSD = plt.psd(data, Fs=Fs, NFFT = NFFT, visible = False)
    freq_dat = np.fft.fftfreq(data.size)*Fs #fourier frequencies corresponding to data partition
    power_spec = np.interp(freq_dat, freq_PSD, power_dat)
    
    df = np.abs(freq_dat[1] - freq_dat[2])
    opt_filter = fft * fft_template.conjugate() / power_spec #optimal filter
    dat_filt = 2*np.fft.ifft(opt_filter) #revert to time domain for filter output
    
    sigmasq = 4*(fft_template * fft_template.conjugate() / power_spec).sum() * df 
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = abs(dat_filt) / (sigma)
    return SNR

# create numpy array with all data
for file in range(1, Nfiles): #Current version of MAGE.vi gives false data in first file
    f = h5py.File(folder + '/' + exp_name + '/' + run_name + '/' + identifier + str(file) + '.hdf5', 'r')
    file_start = t_start + timedelta(seconds = file*Nsample*dt)
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*9.86e-10
            dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*9.86e-10
            
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
    from lmfit.models import LorentzianModel, LinearModel
    
    Q_array = np.zeros((Ninputs, Nchannels, 2))
    Gamma_array = np.zeros(Q_array.shape)
    fcenter_array = np.zeros(Q_array.shape)
    error_array = np.zeros(Q_array.shape)
    Gamma_max = 20.0 #Maximum accepted linewidth in Hz
    error_max = 0.2 #Maximum accepted standard fit error in Hz
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            Sx_trim = Sx[AI, channel, :]
            Sy_trim = Sy[AI, channel, :]
            fn_trim = fn    
    
            peak_model = LorentzianModel()
            noise_model = LinearModel()
    
            pars = peak_model.guess(Sx_trim, x=fn_trim)
            pars += noise_model.guess(Sx_trim, x=fn_trim)
            pars2 = peak_model.guess(Sx_trim, x=fn_trim)
            pars2 += noise_model.guess(Sx_trim, x=fn_trim)
            model = peak_model + noise_model
    
            out = model.fit(Sx_trim, pars, x=fn_trim)
            out2 = model.fit(Sy_trim, pars2, x=fn_trim)
            Gamma1, Gamma2 = out.params["sigma"].value, out2.params["sigma"].value
            fcenter1, fcenter2 = out.params["center"].value, out2.params["center"].value
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
            fcenter_array[AI, channel, :] = [fcenter1, fcenter2]
            error_array[AI, channel, :] = [error1, error2]
            if Gamma1 > Gamma_max or Gamma2 > Gamma_max or error1 > error_max or error2 > error_max: #Ignore bad fits
                print("Input AI " + str(AI) + ", Channel " + str(channel+1) + ":WARNING: Bad mode detected, channel will be ignored")

            # plt.ion()
            # fig = plt.figure("IMPA DOWNLOAD")
            # plt.axis('tight')
            # plt.pause(0.05)
            # plt.draw()
            # fig.clf()
            # ax = fig.add_subplot(111)
            # ax.plot(fn_trim, Sx_trim, 'o', markersize=0.2)
            # ax.set_title("Input AI " + str(AI) + ", Channel " + str(channel+1))
            # #plt.plot(fn_n, out.init_fit, '--', label='initial fit')
            # ax.plot(fn_trim, out.best_fit, '-', label='best fit I')
            # ax.plot(fn_trim, out2.best_fit, '-', label='best fit Q')
            # ax.set_yscale('log')
            # #ax.set_xscale('log')
            # ax.legend()
    
    
    #Filtering data
    Nfilter=NFFT
    tn = np.linspace(0,Nsample*dt,Nsample)
    t_sig = dt*np.linspace(0, Nfilter, Nfilter)
    Φ = 3*np.pi/4


    for AI in range(Ninputs):
        for channel in range(Nchannels):

            f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
            tau1, tau2 = Q_array[AI, channel]/(np.pi*f_demod)
            fpeak_sig = fcenter_array[AI, channel, 0]
            
            if any(Gamma_array[AI, channel, :] > Gamma_max) or any(error_array[AI, channel, :] > error_max): # Skip channels for which a mode within parameters could not be found
              continue
            G = squid_model[AI][0]/np.sqrt(1+(f_demod/squid_model[AI][1])**2) #squid gain
            VtoT = (1/G)**2*resistances[channel][AI]*tau1/kb/4
            
            Idat, Qdat = data_array[AI, channel, 0], data_array[AI, channel, 1] # Phase and quadrature data to be filtered
            
            Rdat = (Idat**2+Qdat**2)*VtoT
            Tn=Rdat
            template = np.exp(-t_sig/(2*tau1)) # template construction
            data = Rdat

            SNR = optimal_filter(data, template, Fs, NFFT)
            threshold = 0.25
            peaks = find_peaks(SNR, height = threshold, distance = int(tau1*Fs), width = [100, 5e6], rel_height=1.0)
            
            # file_start_date = t_start + timedelta(seconds = file*Nsample*dt)
            event_day_string = datetime.strftime(file_start, "%d-%m-%y")
            
            ## data quality cuts
            if len(peaks[0])>0:
                diverge_template1 = np.exp(-t_sig/(tau1/10.0))
                diverge_template2 = np.exp(-t_sig/(tau1/100.0))
                transient_SNR1 = optimal_filter(data, diverge_template1, Fs, NFFT)           
                transient_SNR2 = optimal_filter(data, diverge_template2, Fs, NFFT)
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
                if (SNR[event_i] < transient_SNR1[event_i] or SNR[event_i] < transient_SNR2[event_i]):
                    print("Transiently divergent feature detected of SNR %1.2f" % (SNR[event_i]) + ", performing quality cut")
                    continue
                print("Large event detected" + '\n')
                print("File " + str(file) + " Input AI " + str(AI) + ", Channel " + str(channel+1))
                event_time = file_start + timedelta(seconds = event_i*dt)
                event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-AI" + str(AI) + "-ch" + str(channel+1) + "-SNR %1.2f" % (SNR[event_i])
                event_catalogue[event_name] = {'time' : event_time, 'SNR' : SNR[event_i], 'input AI' : AI, 'channel' : channel+1, 'frequency' : f_demod, 'noise' : np.mean(Tn), 'file N' : file, 'index' : event_i}
                # ax.plot(event_i*dt,SNR[event_i], linestyle = ' ', marker = 'x', color = 'black')
                # if os.path.exists(output_path + '/' + event_day_string) == False:
                #     os.makedirs(output_path+ '/' + event_day_string)
                # pp = output_path + '/' + event_day_string  +'/filteredSignal-%1.2f-AI ' % (SNR[event_i]) + str(AI) + '-channel ' + str(channel+1) +'-' + datetime.strftime(event_time, "%Hp%Mp%S") + ".pdf"
                # plt.savefig(pp, format='pdf', dpi=300)
                print('Event ' + event_name + ' Saved')
               
import pickle
with open(output_path + '/event_catalogue.pkl', 'wb') as f:
    pickle.dump(event_catalogue, f)              
    ## Plot filtered results / mode temperatures --> wont be accurate for MAGE0 Data
    
    ## Transient search for Impulse events some decay shape (N consecutive samples with T>?)
    
    ## 
