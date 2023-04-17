import h5py
from scipy import signal
from scipy.signal import welch, csd
import matplotlib
import matplotlib.pyplot as plt
import os
from os import listdir
import numpy as np
from scipy import fft
import lmfit
matplotlib.use('TkAgg')
from datetime import datetime
from datetime import timedelta
#file by file data analysise

## Transient filter search for MAGE data stream. First test designed for MAGE0 Data

#load data file
folder = r"C:\Users\21958742\DarkMatterCentre Dropbox\William Campbell\PhD\High Frequency GW\Data Analysis\MAGE0/"
exp_name = "MAGE0 v2"
run_name = "run1"

SNR_threshold = 40

files = listdir(folder + '/' + exp_name + '/' + run_name)
Nfiles = len(files)

#meta data from first file
f = h5py.File(folder + '/' + exp_name + '/' + run_name + '/' + files[0], 'r')

Ninputs = len(f.keys())
Nchannels = len(f['AI 0'].keys())//2
Nsample = len(f['AI 0/CH 1-I/Data'][:])
Fs = f['AI 0'].attrs['Fs']
data_array = np.zeros((Ninputs, Nchannels, 2, Nsample))
t_start_string = str(f['AI 0'].attrs['date/time string'])
t_start = datetime.strptime(t_start_string, "b'UTC %d-%m-%y %H:%M:%S.%f '")


#create output folder to store analysis results
output_path = folder + '/' + exp_name + '/Analysis/' + run_name
if os.path.exists(output_path) == False:
    os.makedirs(output_path)

# create numpy array with all data
for file in range(1, Nfiles): #Current version of MAGE.vi gives false data in first file
    f = h5py.File(folder + '/' + exp_name + '/' + run_name + '/' + files[file], 'r')
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*9.86e-10
            dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*9.86e-10
            
            data_array[AI, channel, 0, :] = dataI
            data_array[AI, channel, 1, :] = dataQ

    #Determine single sided power spectrum for each stream
    NFFT = 2**10 # for NFFT < Nsample power spectrum will be averaged
    
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
            f_lockin = f['AI '+str(AI)].attrs['Demod freqs AI '+str(AI)][channel]
    
            Q1, Q2 = (fcenter1+f_lockin)/(2*Gamma1), (fcenter2+f_lockin)/(2*Gamma2)
            Q_array[AI, channel, :] = [Q1,Q2]
            Gamma_array[AI, channel, :] = [Gamma1,Gamma2]
            fcenter_array[AI, channel, :] = [fcenter1, fcenter2]
            
            if Gamma1 > 1 or Gamma2 > 1:
                print("Input AI " + str(AI) + ", Channel " + str(channel) + ": Bad mode detected, channel will be ignored")
            plt.ion()
            fig = plt.figure("IMPA DOWNLOAD")
            plt.axis('tight')
            plt.pause(0.05)
            plt.draw()
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(fn_trim, Sx_trim, 'o')
            ax.set_title("Input AI " + str(AI) + ", Channel " + str(channel))
            #plt.plot(fn_n, out.init_fit, '--', label='initial fit')
            ax.plot(fn_trim, out.best_fit, '-', label='best fit I')
            ax.plot(fn_trim, out2.best_fit, '-', label='best fit Q')
            ax.legend()
    
    
    #Filtering data
    Nfilter=2**14
    dt = 1/Fs
    tn = np.linspace(0,Nsample*dt,Nsample)
    t_sig = dt*np.linspace(0, Nfilter, Nfilter)
    Φ = 3*np.pi/4
    
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            if Gamma_array[AI, channel, 0] > 1 or Gamma_array[AI, channel, 1] > 1: # Skip channels for which a mode of fwhm < 1 Hz could not be found
              continue
            Idat, Qdat = data_array[AI, channel, 0], data_array[AI, channel, 1] # Phase and quadrature data to be filtered
            f_demod = f['AI ' + str(AI)].attrs['Demod freqs AI ' + str(AI)][channel] # demodulation frequency
            
            templateI = np.exp(-t_sig/2.0)*np.cos(2*np.pi*(fcenter_array[AI, channel, 0])*t_sig + Φ) # template construction
            templateQ = np.exp(-t_sig/2.0)*np.sin(2*np.pi*(fcenter_array[AI, channel, 1])*t_sig + Φ)
            
            fftI , fftQ = np.fft.fft(Idat), np.fft.fft(Qdat) # fourier transformed data
            zero_pad = np.zeros(Idat.size - templateI.size) # zero pad template to match data size
            template_padI, template_padQ = np.append(templateI, zero_pad), np.append(templateQ, zero_pad)
            fft_templateI, fft_templateQ = np.fft.fft(template_padI), np.fft.fft(template_padQ) # fourier transformed padded template
            
            
            freq_dat = np.fft.fftfreq(Qdat.size)*Fs #fourier frequencies corresponding to data partition
            power_specI, power_specQ = np.interp(freq_dat, fn, Sx[AI, channel]), np.interp(freq_dat, fn, Sy[AI, channel]) #interpolated PSD 
            
            df = np.abs(freq_dat[1] - freq_dat[2])
            opt_filterI, opt_filterQ = fftI * fft_templateI.conjugate() / power_specI, fftQ * fft_templateQ.conjugate() / power_specQ #optimal filter
            Idat_filt, Qdat_filt = 2*np.fft.ifft(opt_filterI), 2*np.fft.ifft(opt_filterQ) #revert to time domain for filter output
            
            sigmasqI, sigmasqQ = 2*(fft_templateI * fft_templateQ.conjugate() / power_specI).sum() * df , 2*(fft_templateQ * fft_templateQ.conjugate() / power_specQ).sum() * df
            sigmaI, sigmaQ = np.sqrt(np.abs(sigmasqI)), np.sqrt(np.abs(sigmasqQ))
            SNRI, SNRQ = abs(Idat_filt) / (sigmaI), abs(Qdat_filt) / (sigmaQ)
            
            plt.ion()
            fig = plt.figure("Filtered Output")
            plt.axis('tight')
            plt.pause(0.05)
            plt.draw()
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_title("File " + str(file+1) + " Input AI " + str(AI) + ", Channel " + str(channel))
            ax.plot(tn, np.sqrt(Idat**2+Qdat**2)/np.std(np.sqrt(Idat**2+Qdat**2)), label = 'raw')
            #ax.plot(tn, np.sqrt(SNRI**2+SNRQ**2)/np.std(np.sqrt(SNRI**2+SNRQ**2)), label = 'filtered')
            ax.plot(tn, np.sqrt(SNRI**2+SNRQ**2), label = 'filtered')
            ax.legend()
            R = np.sqrt(SNRI**2+SNRQ**2)
            if any([Tn for Tn in R > SNR_threshold]):
                print("Large event detected" + '\n')
                print("File " + str(file+1) + " Input AI " + str(AI) + ", Channel " + str(channel))
                
                file_start_date = t_start + timedelta(seconds = file*Nsample*dt)
                event_day_string = datetime.strftime(file_start_date, "%d-%m-%y")
                event_time_string = datetime.strftime(file_start_date, "%Hp%Mp%S")
                if os.path.exists(output_path + '/' + event_day_string) == False:
                    os.makedirs(output_path+ '/' + event_day_string)
                pp = output_path + '/' + event_day_string  +'/filteredSignal-%1.2f-AI ' % (np.max(R)) + str(AI) + '-channel ' + str(channel) +'-' + event_time_string + ".pdf"
                plt.savefig(pp, format='pdf', dpi=600)
                
    
    ## Plot filtered results / mode temperatures --> wont be accurate for MAGE0 Data
    
    ## Transient search for Impulse events some decay shape (N consecutive samples with T>?)
    
    ## 