#TODO: Separate codebase into modules, define different experiment types
from importlib.resources import files, path
import dataStream
import EventSearchUtils
from typing import Optional, List, Dict, Callable
import re
import pickle
from os import listdir
import numpy as np
import h5py
from datetime import datetime, timedelta
from scipy.signal import welch, find_peaks
from lmfit.models import LorentzianModel, ConstantModel
import matplotlib.pyplot as plt
import Analysis_functions
import os
from itertools import combinations
from natsort import natsorted

#constants
kb = 1.380649e-23 #Boltzmann constant
phi0 = 2.067833848e-15 #Flux quantum

class Experiment:
    """Container for multiple Runs (each run contains multiple DataFile instances)."""
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.runs: List[dataStream.Run] = []

    def add_run(self, run: dataStream.Run):
        self.runs.append(run)
    
    #TODO Load runs from folder structure

    def iter_runs(self):
        return iter(self.runs)

    def __len__(self):
        return len(self.runs)  
#TODO: Place matched filter search in its own module
def filter_search_on_data(run:dataStream.Run, avoid_files = [], show_plot=False, simulate_with_noise=False, do_coincident_analysis=False, **kwargs):
    _file_names = natsorted(listdir(run._filepath))
    _Nfiles = len(_file_names)
    mode_calibration_data = run.mode_calibration_data
    squid_calibration_data = run.squids
    _identifier = kwargs.pop('identifier', None)
    _NFFT = kwargs.pop('NFFT', 2**12)
    _scaling_gain = kwargs.pop('scaling_gain',3.14e-5)
    _SNR_threshold = kwargs.pop('SNR_threshold', 1.0)
    Gamma_max = 20.0 #Maximum accepted linewidth in Hz
    error_max = 0.5 #Maximum accepted standard fit error in Hz
    #TODO: Will throw error if metadata not updated.
    Fs = run.metadata['sample_rate']
    Nsample = run.metadata['N_samples']
    event_catalogue={}
    candidate_events = []
    #create output folder to store analysis results
    output_path = run._output_path
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    if _identifier is None:
        _identifier = run.name + '_' #run1_1.hdf5 etc...
    for _file_index, _file_name in enumerate(_file_names): #Current version of MAGE.vi gives false data in first file
        if _file_index in avoid_files: #avoid files
            continue
        #Setup for optimal filtering 
        Nfilter=_NFFT
        tn = np.linspace(0,Nsample*1/Fs,Nsample)
        t_sig = 1/Fs*np.linspace(0, Nfilter, Nfilter)
        event_catalogue_perfile = {}
        #Read in local file data temporarily. i.e. does not save or read data to dataStream class.
        f = h5py.File(run._filepath + '/' + _identifier + str(_file_index) + '.hdf5', 'r')
        file_start_string =  f['AI 0'].attrs['date/time string']
        file_start = datetime.strptime(file_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
        for _AI, _detector_name in enumerate(run.detectors_in_run()):
            #detector calibration data
            if _detector_name not in squid_calibration_data:
                raise ValueError("SQUID Calibration data for detector " + _detector_name + " ensure SQUID is initiated.")
                break
            Vphi = squid_calibration_data[_detector_name].Vphi
            Min = squid_calibration_data[_detector_name].Min
            for _channel, _channel_name in enumerate(run.channels_in_detector(_detector_name)):
                #TODO Automate naming conventions
                dataI = f['AI ' + str(_AI) + '/CH ' + str(_channel+1) + '-I/Data'][:]*_scaling_gain**2
                dataQ = f['AI ' + str(_AI) + '/CH ' + str(_channel+1) + '-Q/Data'][:]*_scaling_gain**2
                f_demod = f['AI ' + str(_AI)].attrs['Demod freqs AI ' + str(_AI)][_channel] 

                #mode calibration data
                meff = mode_calibration_data._channel_name['Meff']
                Rlambda = mode_calibration_data._channel_name['Rlambda']
                xi = mode_calibration_data._channel_name['xi']

                fn, SdataI = welch(dataI, fs=Fs, nperseg = 2*_NFFT-1, scaling = 'density')
                fn, SdataQ = welch(dataQ, fs=Fs, nperseg = 2*_NFFT-1, scaling = 'density')

                fit_result_I = Analysis_functions.lorentzian_fit_thermalpeak_bis_onlyFandQ(SdataI, fn, f_demod, _AI, _channel, Plot=show_plot, span=300)
                fit_result_Q = Analysis_functions.lorentzian_fit_thermalpeak_bis_onlyFandQ(SdataQ, fn, f_demod, _AI, _channel, Plot=show_plot, span=300)
        
                print(_file_name)


                Q1, Q2 = fit_result_I['Q_factor'], fit_result_Q['Q_factor']
                Gamma1, Gamma2 = fit_result_I['linewidth'], fit_result_Q['linewidth']
                error1, error2 = fit_result_I['linewidth_error'], fit_result_Q['linewidth_error']
            
                if (Gamma1 > Gamma_max and Gamma2 > Gamma_max) or (error1 > error_max and error2 > error_max): #Ignore bad fits
                    print("Input AI " + str(_AI) + ", Channel " + str(_channel+1) + ":WARNING: Bad mode detected, channel will be ignored")
                    continue

                tau1, tau2 = np.array([Q1,Q2])/(np.pi*f_demod)
                #TODO add input for intrinsic Q-factors, these ones are inferred from PSD fit.
                
                if simulate_with_noise:
                    dataI_noise = np.random.normal(np.mean(dataI), np.std(dataI), size = len(dataI))
                    dataQ_noise = np.random.normal(np.mean(dataQ), np.std(dataQ), size = len(dataQ))
                    Rdat = np.sqrt(dataI_noise**2+dataQ_noise**2)
                else:    
                    Rdat = np.sqrt(dataI**2+dataQ**2)
                idx = (np.abs(np.array([tau1,tau2]) - 1.0)).argmin()
                tau = np.array([tau1,tau2])[idx]
                Nfilter=int(Fs*5*tau)
                tn = np.linspace(0,Nsample*1/Fs,Nsample)
                t_sig = 1/Fs*np.linspace(0, Nfilter, Nfilter)
                #calibration
                G = (Vphi*2000)*Min
                kappa = np.sqrt(f_demod*2*np.pi*meff/(np.mean([Q1,Q2]) * Rlambda))
                h = np.fft.ifft(np.fft.fft(Rdat[1:]/G)/(kappa*f_demod*2*np.pi)).real
                # template construction
                template = np.exp(-t_sig/(tau))
                T = np.mean(0.5*(f_demod*2*np.pi)**2*meff*h**2/kb)
                SNR, dat_filt = Analysis_functions.optimal_filter(h, template, Fs, _NFFT)
                
                _SNR_detection_threshold = 1.0 ## effective noise temperature
                peaks = find_peaks(SNR, height = _SNR_detection_threshold, distance = int(tau*Fs), width = [100, 5e6], rel_height=1.0)
                # file_start_date = t_start + timedelta(seconds = file*Nsample*dt)
                event_day_string = datetime.strftime(file_start, "%d-%m-%y")
                
                ## data quality cuts
                if len(peaks[0])>0:
                    diverge_template1 = np.exp(-t_sig/(tau/10.0))
                    diverge_template2 = np.exp(-t_sig/(tau*10.0))
                    transient_SNR1, junk = optimal_filter(h, diverge_template1, Fs, _NFFT)           
                    transient_SNR2, junk = optimal_filter(h, diverge_template2, Fs, _NFFT)
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
                        #print("Transiently divergent feature detected of SNR %1.2f" % (SNR[event_i]) + ", performing quality cut")
                        continue
                    #print("Large event detected" + '\n')
                    print("File " + str(_file_index) + _detector_name + str(_AI) + ", Channel " + str(_channel+1))
                    event_time = file_start + timedelta(seconds = event_i*dt)
                    event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-Det " + _detector_name + "-ch" + str(_channel+1) + "-SNR %1.2f" % (SNR[event_i])
                    if event_name not in event_catalogue: 
                        event_catalogue[event_name] = {'time' : event_time, 'SNR' : SNR[event_i],'Teff' : T, 'detector' : _detector_name, 'channel' : _channel_name, 'frequency' : f_demod, 'amplitude' : dat_filt[event_i], 'file N' : _file_index, 'index' : event_i}
                        print('Event ' + event_name + ' Saved')
                    if event_name not in event_catalogue_perfile:
                        event_catalogue_perfile[event_name] = {'time' : event_time, 'SNR' : SNR[event_i],'Teff' : T, 'detector' : _detector_name, 'channel' : _channel_name, 'frequency' : f_demod, 'amplitude' : dat_filt[event_i], 'file N' : _file_index, 'index' : event_i}
        # Coincident modes on one file
        if do_coincident_analysis:
            print("Looking for Coincident events...")
            #TODO this is hairy, takes previous known call to _detecotr_name.
            for _channel, _channel_name in enumerate(run.channels_in_run()):
                channel_trigger_times ={}
                for _AI, _detector_name in enumerate(run.detectors_in_run()):
                    channel_trigger_times[_detector_name] = [event_catalogue_perfile[event]['time'].timestamp() for event in event_catalogue_perfile if 
                            (event_catalogue_perfile[event]['detector'] == _detector_name) and (event_catalogue_perfile[event]['channel'] == _channel_name)]
                detector_pairs = list(combinations(run.detectors_in_run(), 2))
                for pair in detector_pairs:
                    coincident_t = []
                    times0 = channel_trigger_times[pair[0]]
                    times1 = channel_trigger_times[pair[1]]
                    for time0 in times0:
                        for time1 in times1:
                            if np.abs(time0-time1) < 3*1/Fs:
                                #print("Coincident Event at " + str(time0))
                                coincident_t.append((time0,time1))
                                continue
                    for ii in range(len(coincident_t)):
                        co_event_nn = ii
                        per_channel_events = [event for event in event_catalogue_perfile if (event_catalogue_perfile[event]['channel'] == _channel_name)]
                        co_event0 = [(event, event_catalogue_perfile[event]) for event in per_channel_events if 
                                    (event_catalogue_perfile[event]['time'] == datetime.fromtimestamp(coincident_t[co_event_nn][0])) and
                                    event_catalogue_perfile[event]['detector']==pair[0]]
                        co_event1 = [(event, event_catalogue_perfile[event]) for event in per_channel_events if 
                                    (event_catalogue_perfile[event]['time'] == datetime.fromtimestamp(coincident_t[co_event_nn][1])) and
                                    event_catalogue_perfile[event]['detector']==pair[1]]
                        for event0 in co_event0:
                            for event1 in co_event1:
                                candidate_events.append([event0,event1])                  
            print("Candidate events :" + str(len(candidate_events)))
        f.close()
    return event_catalogue, candidate_events


#This class stores the output of a matched filter search.
class FilterSearch(Experiment):
    def __init__(self, output_directory: str, search_runs: List[dataStream.Run], name: Optional[str] = None):
        super().__init__(name)
        #directory to store search result (pickle files)
        self.output_directory = output_directory
        for _run in search_runs:
            if _run not in self.runs:
                raise ValueError(f"Run '{_run.name}' not found in this experiment. Please add the run to the experiment before performing the search.")
        search_results = filter_search_on_data(run_directory, _run)
        
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


    