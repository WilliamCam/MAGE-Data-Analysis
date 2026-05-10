#TODO: Separate codebase into modules, define different experiment types
from os import listdir
import experiment
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import sys
from itertools import combinations
from natsort import natsorted

#constants
kb = 1.380649e-23 #Boltzmann constant
phi0 = 2.067833848e-15 #Flux quantum

def ascii_progress_bar(current, total, width=50, prefix="Progress", warning=""):
    """Create an ASCII progress bar that updates on a single line."""
    percent = float(current) / total
    filled = int(width * percent)
    bar = '█' * filled + '-' * (width - filled)
    status = f"{prefix}: [{bar}] {current}/{total} ({percent:.1%})"
    if warning:
        status += f" | {warning}"
    return f"\r{status}"

def update_status_line(message):
    """Update a status message on a single line."""
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()

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

class FilterSearch(experiment.Experiment):
    def __init__(self, name, master_filepath, config_yaml, read_metadata_on_init=True, **kwargs):
        super().__init__(name, master_filepath, config_yaml, read_metadata_on_init, **kwargs)
        self.event_catalogue={}

    def search_all_files(self, run:experiment.Run, avoid_files = [], show_plot=False, simulate_with_noise=False, do_coincident_analysis=False, **kwargs):
        _file_names = natsorted(listdir(run.filepath))
        _identifier = kwargs.pop('identifier', None)
        Gamma_bounds =  kwargs.pop('Gamma_bounds', [1.0,20.0])
        error_max =  kwargs.pop('error_max', 5.0)
        _NFFT = kwargs.pop('NFFT', 2**12)
        Nfilter=_NFFT
        _SNR_threshold = kwargs.pop('SNR_threshold', 1.0)
        if not run.parent:
            raise ValueError("Run has no parent Experiment")
        exp = self
        squids = exp.squids
        crystals = exp.crystals
        event_catalogue={}
        candidate_events = []
        #create output folder to store analysis results
        output_path = exp._output_path
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)

        if _identifier is None:
            _identifier = run.name + '_' #run1_1.hdf5 etc...
        
        # ASCII progress bar for files
        total_files = len([f for i, f in enumerate(_file_names) if i not in avoid_files])
        processed_files = 0
        
        for _file_index, _file_name in enumerate(_file_names):
            event_catalogue_perfile = {}
            if _file_index in avoid_files: #avoid files
                continue
            
            processed_files += 1
            sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files"))
            sys.stdout.flush()

            _datafile = run.files[_file_index]
            
            exp.load_file(_datafile)

            metadata = exp.metadata[run.name]
            file_start_string =  _datafile.metadata[list(_datafile.metadata.keys())[0]]['date/time string']
            file_start = datetime.strptime(file_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
            file_name = os.path.basename(_datafile.filepath)
            
            file_meta = metadata['Files'][file_name]
            for _AI, (_detector_name, _detector_meta) in enumerate(file_meta['Detectors'].items()):
                if _detector_name not in squids:
                    raise ValueError("SQUID Calibration data for detector " + _detector_name + " not found, ensure SQUID is initiated.")
                Fs = metadata['Attributes'][_detector_name]['Fs']
                G = squids[_detector_name].squid_gain()
                for _channel_name in _detector_meta['Channels']:
                    #Load channel data
                    # print(f"loading channel {_detector_name}, {_channel_name} from file {file_name}")
                    active_channel = _datafile.detectors[_detector_name].channels[_channel_name]
                    #mode calibration data
                    mode_name = exp.config[_detector_name][_channel_name]
                    mode_cal_data = crystals[_detector_name].calibration_data[mode_name]
                    Rlambda = mode_cal_data['Rlambda']
                    meff = mode_cal_data['Meff']
                    f_demod = mode_cal_data['frequency']
                    #fit noise peak and then clear data from RAM
                    fit_result_I, fit_result_Q = active_channel.fit_lorentzian(fs=Fs, nfft=_NFFT, fdemod=f_demod, Plot=show_plot)

                    Q1, Q2 = fit_result_I['Q_factor'], fit_result_Q['Q_factor']
                    Gamma1, Gamma2 = fit_result_I['linewidth'], fit_result_Q['linewidth']
                    error1, error2 = fit_result_I['linewidth_error'], fit_result_Q['linewidth_error']

                    if (not any([Gamma_bounds[0]<Gamma1<Gamma_bounds[1], Gamma_bounds[0]<Gamma2<Gamma_bounds[1]])) or (not any([error1<error_max, error2<error_max])):
                        warning_msg = f"Skipping {_detector_name}:{_channel_name} (bad fit)"
                        sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=warning_msg))
                        sys.stdout.flush()
                        continue

                    strain, T = active_channel.calibrate_strain(f_demod, np.max([Q1,Q2]), Rlambda, meff, G)
                    #active_channel.clear_data()

                    tau1, tau2 = np.array([Q1,Q2])/(np.pi*f_demod)
                    idx = (np.abs(np.array([tau1,tau2]) - 1.0)).argmin()
                    tau = np.array([tau1,tau2])[idx]
                    #TODO add input for intrinsic Q-factors, these ones are inferred from PSD fit.

                    # template construction
                    Nfilter=int(Fs*5*tau)
                    t_sig = 1/Fs*np.linspace(0, Nfilter, Nfilter)
                    template = np.exp(-t_sig/(tau))
                    SNR, dat_filt = optimal_filter(strain, template, Fs, _NFFT)

                    _SNR_detection_threshold = _SNR_threshold ## effective noise temperature
                    peaks = find_peaks(SNR, height = _SNR_detection_threshold, distance = int(3*tau*Fs), width = [100, 5e6], rel_height=1.0)
                    ## data quality cuts
                    if len(peaks[0])>0:
                        diverge_template1 = np.exp(-t_sig/(tau/10.0))
                        diverge_template2 = np.exp(-t_sig/(tau*10.0))
                        transient_SNR1, _ = optimal_filter(strain, diverge_template1, Fs, _NFFT)           
                        transient_SNR2, _ = optimal_filter(strain, diverge_template2, Fs, _NFFT)

                    for event_i in peaks[0]:
                        if ((SNR**2)[event_i] < (transient_SNR1**2)[event_i] or (SNR**2)[event_i] < (transient_SNR2**2)[event_i]):
                            continue
                        event_time = file_start + timedelta(seconds = event_i/Fs)
                        event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-Det " + _detector_name + "-ch" + str(_channel_name) + "-SNR %1.2f" % (SNR[event_i])
                        event_info = {'time' : event_time, 'SNR' : SNR[event_i],'Teff' : T, 'detector' : _detector_name, 'channel' : _channel_name, 'frequency' : f_demod, 'amplitude' : dat_filt[event_i], 'file N' : _file_index, 'index' : event_i}
                        if event_name not in event_catalogue: 
                            event_catalogue[event_name] = event_info
                            active_channel.events[event_name] = event_info
                        if event_name not in event_catalogue_perfile:
                            event_catalogue_perfile[event_name] = event_info

            # Empty channel data from memory
            _datafile.clear_channels()
            
            # Coincident modes on one file
            if do_coincident_analysis:
                sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning="Analyzing coincident events"))
                sys.stdout.flush()
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
                sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=f"Found {len(candidate_events)} coincident events"))
                sys.stdout.flush()
        
        # Clear the progress line and print final summary
        print()  # New line after progress bar
        print(f"Analysis complete: {len(event_catalogue)} total events detected")
        self.event_catalogue = event_catalogue
        return event_catalogue, candidate_events
    
    def inspect_event(self, channel, event_name, span = 1000, _NFFT = 2**12, Qi = None):
        if not channel.events[event_name]:
            raise ValueError("Event does not exist")
        if not channel.data:
            raise ValueError("Data not initalised in Channel")
        event_index = channel.events[event_name]['index']
        _detector_name = channel.parent.name
        _channel_name = channel.name
        _run_name = channel.parent.parent.parent.name
        mode_name = self.config[_detector_name][_channel_name]
        mode_cal_data = self.crystals[_detector_name].calibration_data[mode_name]
        Rlambda = mode_cal_data['Rlambda']
        meff = mode_cal_data['Meff']
        f_demod = mode_cal_data['frequency']
        Fs = self.metadata[_run_name]['Attributes'][_detector_name]['Fs']
        if not Qi:
            Qi = mode_cal_data['Qi']
        G = self.squids[_detector_name].squid_gain()
        tau = np.array(Qi)/(np.pi*f_demod)
        Nfilter=int(Fs*5*tau)
        t_sig = 1/Fs*np.linspace(0, Nfilter, Nfilter)
        template = np.exp(-t_sig/(tau))
        if not channel.is_IQ:
            dataI = channel.data['I']
            dataQ = channel.data['Q']
        strain, _ = channel.calibrate_strain(f_demod, Qi, Rlambda, meff, G)
        SNR, dat_filt = optimal_filter(strain, template, Fs, _NFFT)
        plt.ion()
        fig = plt.figure()
        plt.axis('tight')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(dataI[event_index-span//2:event_index+span//2]/ np.sqrt(np.mean(dataI**2)))
        ax.plot(dataQ[event_index-span//2:event_index+span//2]/np.sqrt(np.mean(dataI**2)))
        ax.plot(SNR[event_index-span//2:event_index+span//2] / np.mean(SNR))
        ax.legend()
        plt.pause(0.05)
        plt.draw()

    def print_events(self):
        """
        Print an ASCII tree showing the number of events and max SNR for each channel.
        """
        print(f"Experiment: {self.name} (Events Summary)")
        if not self.runs:
            print("└── (no runs)")
            return
        for i, run in enumerate(self.runs):
            # Choose connector based on position
            connector = "└──" if i == len(self.runs) - 1 else "├──"
            print(f"{connector} Run: {run.name}")
            
            if not run.files:
                # No files in this run
                continue_connector = "    └──" if i == len(self.runs) - 1 else "    ├──"
                print(f"{continue_connector} (no files)")
                continue
            
            # Print files for this run
            for j, datafile in enumerate(run.files):
                # Choose file connector
                file_connector = "    └──" if j == len(run.files) - 1 else "    ├──"
                run_connector = "    " if i == len(self.runs) - 1 else "│   "
                
                # Extract just the filename from the full path
                filename = os.path.basename(datafile.filepath)
                print(f"{run_connector}{file_connector} {filename}")

                # Show detectors and channels with events
                detector_prefix = run_connector + ("    " if j == len(run.files) - 1 else "│   ")
                if not datafile.detectors:
                    continue

                for k, detector in enumerate(datafile.detectors.values()):
                    detector_connector = "└──" if k == len(datafile.detectors) - 1 else "├──"
                    print(f"{detector_prefix}{detector_connector} Detector: {detector.name}")

                    if not detector.channels:
                        continue

                    # Get channels that have events
                    channels_with_events = [ch for ch in detector.channels.values() if ch.events]
                    if not channels_with_events:
                        continue

                    channel_prefix = detector_prefix + ("    " if k == len(datafile.detectors) - 1 else "│   ")
                    for l, channel in enumerate(channels_with_events):
                        channel_connector = "└──" if l == len(channels_with_events) - 1 else "├──"
                        num_events = len(channel.events)
                        # Find max SNR
                        max_snr = max(event['SNR'] for event in channel.events.values()) if channel.events else 0
                        # Color max SNR: red if > 5, blue if > 3
                        snr_str = f"{max_snr:.2f}"
                        if max_snr > 5:
                            snr_str = f"\033[31m{snr_str}\033[0m"
                        elif max_snr > 3:
                            snr_str = f"\033[34m{snr_str}\033[0m"
                        print(f"{channel_prefix}{channel_connector} Channel: {channel.name} ({num_events} events, max SNR: {snr_str})")