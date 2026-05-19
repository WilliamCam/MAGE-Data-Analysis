#TODO: Separate codebase into modules, define different experiment types
from os import listdir
import experiment
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks, welch
import matplotlib.pyplot as plt
import os
import sys
from itertools import combinations
from natsort import natsorted
import pickle
import gc
from matplotlib.mlab import psd

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
    fft = np.fft.fft(data)
    zero_pad = np.zeros(data.size - template.size)
    template_pad = np.append(template, zero_pad)
    fft_template = np.fft.fft(template_pad)
    power_dat, freq_PSD = psd(data, Fs=Fs, NFFT=NFFT)
    freq_dat = np.fft.fftfreq(data.size) * Fs
    power_spec = np.interp(freq_dat, freq_PSD, power_dat)
    val_cal = np.max(template)
    OF = np.fft.ifft(fft_template * fft_template.conjugate() / power_spec).real
    K = val_cal / np.amax(OF)
    df = np.abs(freq_dat[1] - freq_dat[2])
    opt_filter = K * fft * fft_template.conjugate() / power_spec
    dat_filt = np.fft.ifft(opt_filter)
    sigmasq = 2 * (K**2 * fft_template * fft_template.conjugate() / power_spec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = np.abs(2 * dat_filt) / (sigma)
    del fft, zero_pad, template_pad, fft_template, power_dat, freq_PSD, freq_dat, power_spec, OF, K, df, opt_filter, sigmasq, sigma
    return SNR, dat_filt

class FilterSearch(experiment.Experiment):
    def __init__(self, name, master_filepath, config_yaml, read_metadata_on_init=True, **kwargs):
        super().__init__(name, master_filepath, config_yaml, read_metadata_on_init, **kwargs)
        self.event_catalogue = {}

    def search_all_files(self, run:experiment.Run, avoid_files=[], show_plot=False, simulate_with_noise=False, do_coincident_analysis=False, output_pkl_dir=None, resume=False, **kwargs):
        _file_names = natsorted(listdir(run.filepath))
        _identifier = kwargs.pop('identifier', None)
        Gamma_bounds = kwargs.pop('Gamma_bounds', [1.0, 20.0])
        error_max = kwargs.pop('error_max', 5.0)
        _NFFT = kwargs.pop('NFFT', 2**12)
        force_all_channels = kwargs.pop('force_all_channels', False)
        _SNR_threshold = kwargs.pop('SNR_threshold', 1.0)
        if not run.parent:
            raise ValueError("Run has no parent Experiment")
        exp = self
        squids = exp.squids
        crystals = exp.crystals
        event_catalogue = {}
        candidate_events = []
        
        if output_pkl_dir is None:
            output_pkl_dir = exp._output_path
        if not os.path.exists(output_pkl_dir):
            os.makedirs(output_pkl_dir)
        
        output_path = exp._output_path
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)

        if _identifier is None:
            _identifier = run.name + '_'
        
        start_file_index = 0
        total_files = len([f for i, f in enumerate(_file_names) if i not in avoid_files])
        processed_files = 0
        Nevents = 0
        
        # Handle resume logic
        if resume:
            pkl_files = natsorted([f for f in os.listdir(output_pkl_dir) if f.startswith(_identifier) and 'file_' in f and f.endswith('.pkl')])
            if pkl_files:
                last_pkl = pkl_files[-1]
                try:
                    file_idx_str = last_pkl.split('file_')[1].split('_events')[0]
                    start_file_index = int(file_idx_str) + 1
                    print(f"Resuming from file index {start_file_index} (last completed: {last_pkl})")
                    event_catalogue = self.load_event_catalogue_from_pickles(output_pkl_dir, _identifier)
                    Nevents = len(event_catalogue)
                except Exception as e:
                    print(f"Warning: Could not parse last file index: {e}")
                    print("Starting fresh analysis")
                    start_file_index = 0
                    event_catalogue = {}
                    Nevents = 0
            else:
                print("No existing pickle files found. Starting fresh.")
                event_catalogue = {}
                Nevents = 0
        else:
            print("Starting fresh analysis (overwriting existing pickle files)")
            event_catalogue = {}
            Nevents = 0
        
        total_files = len([f for i, f in enumerate(_file_names) if i not in avoid_files and i >= start_file_index])
        
        for _file_index, _file_name in enumerate(_file_names):
            if _file_index < start_file_index:
                continue
            
            event_catalogue_perfile = {}
            event_catalogue_perfile.clear()
            if _file_index in avoid_files:
                continue
            
            processed_files += 1
            sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files"))
            sys.stdout.flush()

            if processed_files%10==0:
                gc.collect()

            _datafile = run.files[_file_index]
            exp.load_file(_datafile)

            metadata = exp.metadata[run.name]
            file_start_string = _datafile.metadata[list(_datafile.metadata.keys())[0]]['date/time string']
            file_start = datetime.strptime(file_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
            file_name = os.path.basename(_datafile.filepath)
            
            file_meta = metadata['Files'][file_name]
            for _AI, (_detector_name, _detector_meta) in enumerate(file_meta['Detectors'].items()):
                if _detector_name not in squids:
                    raise ValueError("SQUID Calibration data for detector " + _detector_name + " not found, ensure SQUID is initiated.")
                Fs = metadata['Attributes'][_detector_name]['Fs']
                G = squids[_detector_name].squid_gain()
                for _channel_name in _detector_meta['Channels']:
                    active_channel = _datafile.detectors[_detector_name].channels[_channel_name]
                    mode_name = exp.config[_detector_name][_channel_name]
                    mode_cal_data = crystals[_detector_name].calibration_data[mode_name]
                    Rlambda = mode_cal_data['Rlambda']
                    meff = mode_cal_data['Meff']
                    f_demod = mode_cal_data['frequency']
                    if not force_all_channels:
                        fit_result_I, fit_result_Q = active_channel.fit_lorentzian(fs=Fs, nfft=_NFFT, fdemod=f_demod, Plot=show_plot)
                        Q1, Q2 = fit_result_I['Q_factor'], fit_result_Q['Q_factor']
                        Q = np.max([Q1, Q2])
                        Gamma1, Gamma2 = fit_result_I['linewidth'], fit_result_Q['linewidth']
                        error1, error2 = fit_result_I['linewidth_error'], fit_result_Q['linewidth_error']
                        if (not any([Gamma_bounds[0]<Gamma1<Gamma_bounds[1], Gamma_bounds[0]<Gamma2<Gamma_bounds[1]])) or (not any([error1<error_max, error2<error_max])):
                            warning_msg = f"Skipping {_detector_name}:{_channel_name} (bad fit)"
                            sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=warning_msg))
                            sys.stdout.flush()
                            del fit_result_I, fit_result_Q
                            continue
                        del fit_result_I, fit_result_Q
                    else:
                        Q = mode_cal_data['Qi']
                    
                    strain, T = active_channel.calibrate_strain(f_demod, Q, Rlambda, meff, G)
                    tau = Q / (np.pi * f_demod)
                    Nfilter = int(Fs * 5 * tau)
                    t_sig = 1 / Fs * np.linspace(0, Nfilter, Nfilter)
                    template = np.exp(-t_sig / tau)
                    SNR, dat_filt = optimal_filter(strain, template, Fs, _NFFT)
                    
                    _SNR_detection_threshold = _SNR_threshold
                    peaks = find_peaks(SNR, height=_SNR_detection_threshold, distance=int(3*tau*Fs), width=[100, 5e6], rel_height=1.0)
                    
                    if len(peaks[0]) > 0:
                        diverge_template1 = np.exp(-t_sig / (tau / 10.0))
                        diverge_template2 = np.exp(-t_sig / (tau * 10.0))
                        transient_SNR1, _ = optimal_filter(strain, diverge_template1, Fs, _NFFT)
                        transient_SNR2, _ = optimal_filter(strain, diverge_template2, Fs, _NFFT)

                        for event_i in peaks[0]:
                            if ((SNR**2)[event_i] < (transient_SNR1**2)[event_i] or (SNR**2)[event_i] < (transient_SNR2**2)[event_i]):
                                continue
                            warning_msg = f"Total Events found: {Nevents}"
                            sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=warning_msg))
                            sys.stdout.flush()
                            event_time = file_start + timedelta(seconds=event_i/Fs)
                            event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S") + "-Det " + _detector_name + "-ch" + str(_channel_name) + "-SNR %1.2f" % (SNR[event_i])
                            event_info = {'time': event_time, 'SNR': SNR[event_i], 'Teff': T, 'detector': _detector_name, 'channel': _channel_name, 'frequency': f_demod, 'amplitude': dat_filt[event_i], 'file N': _file_index, 'index': event_i}
                            #if event_name not in event_catalogue:
                                #event_catalogue[event_name] = event_info
                                #active_channel.events[event_name] = event_info
                            if event_name not in event_catalogue_perfile:
                                event_catalogue_perfile[event_name] = event_info
                                Nevents += 1
            del _datafile, transient_SNR1, transient_SNR2, strain, SNR, dat_filt, template, t_sig, T, active_channel                             
            #_datafile.clear_channels()
            
            if do_coincident_analysis:
                sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning="Analyzing coincident events"))
                sys.stdout.flush()
                for _channel, _channel_name in enumerate(run.channels_in_run()):
                    channel_trigger_times = {}
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
                                    coincident_t.append((time0, time1))
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
                                    candidate_events.append([event0, event1])                  
                sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=f"Found {len(candidate_events)} coincident events"))
                sys.stdout.flush()
            
            if event_catalogue_perfile:
                pkl_filename = f"{_identifier}file_{_file_index:03d}_events.pkl"
                pkl_path = os.path.join(output_pkl_dir, pkl_filename)
                try:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(event_catalogue_perfile, f)
                    # sys.stdout.write(ascii_progress_bar(processed_files, total_files, prefix="Processing files", warning=f"Saved {pkl_filename}"))
                    # sys.stdout.flush()
                except Exception as e:
                    print(f"\nWarning: Failed to save {pkl_path}: {e}")
            
            event_catalogue_perfile.clear()

        
        if event_catalogue:
            pkl_filename = f"{_identifier}event_catalogue.pkl"
            pkl_path = os.path.join(output_pkl_dir, pkl_filename)
            try:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(event_catalogue, f)
                print(f"\nFull catalogue saved to: {pkl_path}")
            except Exception as e:
                print(f"\nWarning: Failed to save full catalogue: {e}")
        
        print()
        print(f"Analysis complete: {len(event_catalogue)} total events detected")
        self.event_catalogue = event_catalogue
        return event_catalogue, candidate_events
    
    def inspect_event(self, run, event_name, span=1000, _NFFT=2**13, 
                      Qi=None, all_channels=False, channels_to_plot=None, diagnostic_plots=False
                      ):
        
        if event_name not in self.event_catalogue:
            raise ValueError(f"Event '{event_name}' not found in catalogue")
        
        event_info = self.event_catalogue[event_name]
        _detector_name = event_info['detector']
        _channel_name = event_info['channel']
        file_index = event_info['file N']
        event_index = event_info['index']
        
        # Find the run and file
        datafile = run.files[file_index]

        _run_name = run.name
        # if str(file_index) not in str(datafile.filepath):
        #     raise ValueError(f"File index {file_index} does not match file path {datafile.filepath}")
        # # Reload file if data is cleared
        if not datafile.detectors:
            self.load_file(datafile)
            print(f"Reloaded file: {datafile.filepath}")
        try:
            if not datafile.detectors[_detector_name].channels[_channel_name].data:
                self.load_file(datafile)
                print(f"Reloaded file: {datafile.filepath}")
        except (KeyError, AttributeError):
            self.load_file(datafile)
            print(f"Reloaded file: {datafile.filepath}")
        
        metadata = self.metadata[_run_name]
        file_meta = metadata['Files'][os.path.basename(datafile.filepath)]
        
        # If all_channels requested, collect channels to plot
        if all_channels:
            if channels_to_plot is None:
                # Plot all channels
                channels_to_plot = []
                for detector_name, detector_meta in file_meta['Detectors'].items():
                    for ch_name in detector_meta['Channels']:
                        channels_to_plot.append((detector_name, ch_name))
            channels_to_plot = [(d, c) for d, c in channels_to_plot]  # ensure list of tuples
        else:
            # Plot only trigger channel
            channels_to_plot = [(_detector_name, _channel_name)]
        
        # Compute SNR for each channel to plot
        all_channels_data = []
        max_snr_overall = 0
        
        for detector_name, channel_name in channels_to_plot:
            try:
                if detector_name not in datafile.detectors or channel_name not in datafile.detectors[detector_name].channels:
                    continue
                
                active_channel = datafile.detectors[detector_name].channels[channel_name]
                # Get calibration data
                mode_name = self.config[detector_name][channel_name]
                mode_cal_data = self.crystals[detector_name].calibration_data[mode_name]
                Rlambda = mode_cal_data['Rlambda']
                meff = mode_cal_data['Meff']
                f_demod = mode_cal_data['frequency']
                if not Qi:
                    Qi_use = mode_cal_data['Qi']
                else:
                    Qi_use = Qi
                
                Fs = metadata['Attributes'][detector_name]['Fs']
                G = self.squids[detector_name].squid_gain()
                I_raw = active_channel.data['I']
                Q_raw = active_channel.data['Q']
            
                # Get calibrated strain
                strain, _ = active_channel.calibrate_strain(f_demod, Qi_use, Rlambda, meff, G)
                # Create template and compute SNR
                tau = Qi_use / (np.pi * f_demod)
                Nfilter = int(Fs * 5 * tau)
                t_sig = 1 / Fs * np.linspace(0, Nfilter, Nfilter)
                template = np.exp(-t_sig / tau)
                SNR, dat_filt = optimal_filter(strain, template, Fs, _NFFT)
                # Extract data around event
                start_idx = max(0, event_index - span // 2)
                end_idx = min(len(SNR), event_index + span // 2)
                
                snr_segment = SNR[start_idx:end_idx]
                i_segment = I_raw[start_idx:end_idx]
                q_segment = Q_raw[start_idx:end_idx]
                # Normalize I and Q by their RMS values
                i_rms = np.sqrt(np.mean(i_segment**2))
                q_rms = np.sqrt(np.mean(q_segment**2))
                i_normalized = i_segment / i_rms if i_rms > 0 else i_segment
                q_normalized = q_segment / q_rms if q_rms > 0 else q_segment
                
                max_snr = np.max(snr_segment)
                max_snr_overall = max(max_snr_overall, max_snr)
                
                all_channels_data.append({
                    'detector': detector_name,
                    'channel': channel_name,
                    'snr': snr_segment,
                    'i_norm': i_normalized,
                    'q_norm': q_normalized,
                    'max_snr': max_snr,
                    'event_idx': span // 2,
                    'is_trigger': (detector_name == _detector_name and channel_name == _channel_name)
                })
            except Exception as e:
                print(f"Warning: Could not process {detector_name}:{channel_name} - {e}")
                continue
        
        if not all_channels_data:
            print(f"No channels found for event {event_name}")
            return
        
        # Create subplot grid
        if len(all_channels_data) == 1:
            num_rows, num_cols = 1, 1
        else:
            num_cols = 2
            num_rows = int(np.ceil(len(all_channels_data) / num_cols))
        
        plt.ion()
        fig = plt.figure(figsize=(14, 4 * num_rows))
        fig.suptitle(f"Event: {event_name}\nTime: {event_info['time']}", fontsize=14, fontweight='bold')
        
        for idx, ch_data in enumerate(all_channels_data, 1):
            ax = fig.add_subplot(num_rows, num_cols, idx)
            
            snr_segment = ch_data['snr']
            i_norm = ch_data['i_norm']
            q_norm = ch_data['q_norm']
            max_snr = ch_data['max_snr']
            event_idx = ch_data['event_idx']
            
            # Plot SNR, I, and Q on same axes
            ax.plot(i_norm, linewidth=1.5, color='orange', alpha=0.7, label='I (normalized)')
            ax.plot(q_norm, linewidth=1.5, color='green', alpha=0.7, label='Q (normalized)')
            ax.plot(snr_segment, linewidth=2, color='steelblue', label='SNR')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Color code by max SNR
            if max_snr > 5:
                title_color = 'red'
            elif max_snr > 3:
                title_color = 'blue'
            else:
                title_color = 'black'
            
            title_str = f"{ch_data['detector']} : {ch_data['channel']}"
            if ch_data['is_trigger']:
                title_str += " (TRIGGER)"
            
            ax.set_title(title_str, color=title_color, fontweight='bold')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Normalized Amplitude / SNR")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.pause(0.05)
        plt.draw()

    def print_events(self):
        """
        Print an ASCII tree showing the number of events and max SNR for each channel.
        Searches self.event_catalogue directly (independent of channel.events state).
        """
        print(f"Experiment: {self.name} (Events Summary)")
        if not self.event_catalogue:
            print("└── (no events in catalogue)")
            return
        
        if not self.runs:
            print("└── (no runs)")
            return
        
        total_events = 0

        for i, run in enumerate(self.runs):
            connector = "└──" if i == len(self.runs) - 1 else "├──"
            print(f"{connector} Run: {run.name}")
            
            if not run.files:
                continue_connector = "    └──" if i == len(self.runs) - 1 else "    ├──"
                print(f"{continue_connector} (no files)")
                continue
            
            for j, datafile in enumerate(run.files):
                file_connector = "    └──" if j == len(run.files) - 1 else "    ├──"
                run_connector = "    " if i == len(self.runs) - 1 else "│   "
                filename = os.path.basename(datafile.filepath)
                
                # Events in this file
                file_events = [name for name, info in self.event_catalogue.items()
                              if info.get('file N') == j]
                
                if not file_events:
                    continue

                print(f"{run_connector}{file_connector} {filename}")

                detector_prefix = run_connector + ("    " if j == len(run.files) - 1 else "│   ")

                # Group events by detector
                detectors_in_file = {}
                for event_name in file_events:
                    det = self.event_catalogue[event_name].get('detector')
                    if det not in detectors_in_file:
                        detectors_in_file[det] = []
                    detectors_in_file[det].append(event_name)

                for k, (detector_name, det_events) in enumerate(detectors_in_file.items()):
                    detector_connector = "└──" if k == len(detectors_in_file) - 1 else "├──"
                    print(f"{detector_prefix}{detector_connector} Detector: {detector_name}")

                    # Group events by channel
                    channels_in_detector = {}
                    for event_name in det_events:
                        ch = self.event_catalogue[event_name].get('channel')
                        if ch not in channels_in_detector:
                            channels_in_detector[ch] = []
                        channels_in_detector[ch].append(event_name)

                    channel_prefix = detector_prefix + ("    " if k == len(detectors_in_file) - 1 else "│   ")
                    for l, (channel_name, ch_events) in enumerate(channels_in_detector.items()):
                        channel_connector = "└──" if l == len(channels_in_detector) - 1 else "├──"
                        num_events = len(ch_events)
                        total_events += num_events
                        max_snr = max(self.event_catalogue[name].get('SNR', 0) for name in ch_events)
                        snr_str = f"{max_snr:.2f}"
                        if max_snr > 5:
                            snr_str = f"\033[31m{snr_str}\033[0m"
                        elif max_snr > 3:
                            snr_str = f"\033[34m{snr_str}\033[0m"
                        print(f"{channel_prefix}{channel_connector} Channel: {channel_name} ({num_events} events, max SNR: {snr_str})")

        print(f"\nTotal events found: {total_events}")
    
    def get_event_names_above_snr(self, threshold: float, run: 'experiment.Run' = None):
        """
        Print ASCII tree and return sorted list of unique event names with SNR >= threshold.
        Searches self.event_catalogue directly (independent of channel.events state).
        If run is provided, filter events from that run only; otherwise use all events.
        Only prints files/detectors/channels that have events above threshold.
        """
        if not self.event_catalogue:
            print("Event catalogue is empty")
            return []

        if run is not None:
            runs_to_search = [run]
            print(f"Events with SNR >= {threshold} in Run: {run.name}")
        else:
            runs_to_search = self.runs
            print(f"Experiment: {self.name} (Events with SNR >= {threshold})")

        if not runs_to_search:
            print("└── (no runs)")
            return []

        event_names = []
        event_count = 0
        run_has_events = False

        for i, r in enumerate(runs_to_search):
            connector = "└──" if i == len(runs_to_search) - 1 else "├──"
            print(f"{connector} Run: {r.name}")
            run_has_events = False

            if not r.files:
                continue_connector = "    └──" if i == len(runs_to_search) - 1 else "    ├──"
                print(f"{continue_connector} (no files)")
                continue

            for j, datafile in enumerate(r.files):
                file_connector = "    └──" if j == len(r.files) - 1 else "    ├──"
                run_connector = "    " if i == len(runs_to_search) - 1 else "│   "
                filename = os.path.basename(datafile.filepath)
                
                # Events in this file
                file_events = [name for name, info in self.event_catalogue.items()
                              if info.get('file N') == j]
                
                if not file_events:
                    continue

                detector_prefix = run_connector + ("    " if j == len(r.files) - 1 else "│   ")

                # Group events by detector
                detectors_in_file = {}
                for event_name in file_events:
                    det = self.event_catalogue[event_name].get('detector')
                    if det not in detectors_in_file:
                        detectors_in_file[det] = []
                    detectors_in_file[det].append(event_name)

                file_has_events = False
                for k, (detector_name, det_events) in enumerate(detectors_in_file.items()):
                    # Group events by channel
                    channels_in_detector = {}
                    for event_name in det_events:
                        ch = self.event_catalogue[event_name].get('channel')
                        if ch not in channels_in_detector:
                            channels_in_detector[ch] = []
                        channels_in_detector[ch].append(event_name)

                    detector_has_events = False
                    channel_prefix = detector_prefix + ("    " if k == len(detectors_in_file) - 1 else "│   ")
                    for l, (channel_name, ch_events) in enumerate(channels_in_detector.items()):
                        # Filter by threshold
                        above_threshold = [name for name in ch_events
                                         if self.event_catalogue[name].get('SNR', 0) >= threshold]
                        
                        if not above_threshold:
                            continue

                        # Only print file and detector lines if they have events above threshold
                        if not file_has_events:
                            print(f"{run_connector}{file_connector} {filename}")
                            file_has_events = True
                            run_has_events = True
                        
                        if not detector_has_events:
                            detector_connector = "└──" if k == len(detectors_in_file) - 1 else "├──"
                            print(f"{detector_prefix}{detector_connector} Detector: {detector_name}")
                            detector_has_events = True

                        channel_connector = "└──" if l == len(channels_in_detector) - 1 else "├──"
                        num_events = len(above_threshold)
                        max_snr = max(self.event_catalogue[name].get('SNR', 0) for name in above_threshold)
                        snr_str = f"{max_snr:.2f}"
                        if max_snr > 5:
                            snr_str = f"\033[31m{snr_str}\033[0m"
                        elif max_snr > 3:
                            snr_str = f"\033[34m{snr_str}\033[0m"
                        print(f"{channel_prefix}{channel_connector} Channel: {channel_name} ({num_events} events, max SNR: {snr_str})")

                        event_names.extend(above_threshold)
                        event_count += num_events

            if not run_has_events:
                print("    └── (no events above threshold)")

        print(f"\nTotal events above SNR {threshold}: {event_count}")
        return sorted(set(event_names))

    def get_largest(self, run: 'experiment.Run' = None, all_channels=False, channels_to_plot=None):
        """
        Find the event with the highest SNR and inspect it.
        If run is provided, search only that run; otherwise search all events.
        
        Parameters:
        - run: optional Run to limit search
        - all_channels: if True, plot all channels at event time
        - channels_to_plot: list of (detector_name, channel_name) tuples to plot
        """
        if not self.event_catalogue:
            print("Event catalogue is empty")
            return None
        
        # Filter events by run if specified
        if run is not None:
            run_file_indices = list(range(len(run.files)))
            events_to_search = {name: info for name, info in self.event_catalogue.items()
                               if info.get('file N') in run_file_indices}
        else:
            events_to_search = self.event_catalogue
        
        if not events_to_search:
            print("No events found")
            return None
        
        # Find event with maximum SNR
        largest_event_name = max(events_to_search.items(), key=lambda x: x[1].get('SNR', 0))[0]
        
        # Inspect the event
        self.inspect_event(run, largest_event_name, all_channels=all_channels, channels_to_plot=channels_to_plot)
        
        return largest_event_name

    def load_event_catalogue_from_pickles(self, pkl_dir: str, identifier: str = None):
        """
        Load event catalogue from pickle files saved by search_all_files.
        Reads both per-file pickles and the full catalogue pickle.
        
        Parameters:
        - pkl_dir: directory containing pickle files
        - identifier: optional identifier prefix to filter files (e.g., "run1_")
                     if None, loads all .pkl files in directory
        
        Returns the merged event_catalogue dict.
        """
        if not os.path.exists(pkl_dir):
            raise ValueError(f"Pickle directory does not exist: {pkl_dir}")
        
        self.event_catalogue = {}
        pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
        
        if not pkl_files:
            print(f"No pickle files found in {pkl_dir}")
            return self.event_catalogue
        
        # Filter by identifier if provided
        if identifier:
            pkl_files = [f for f in pkl_files if f.startswith(identifier)]
        
        if not pkl_files:
            print(f"No pickle files with identifier '{identifier}' found in {pkl_dir}")
            return self.event_catalogue
        
        # Try to load full catalogue first
        full_catalogue_files = [f for f in pkl_files if 'event_catalogue.pkl' in f]
        if full_catalogue_files:
            full_pkl_path = os.path.join(pkl_dir, full_catalogue_files[0])
            try:
                with open(full_pkl_path, 'rb') as f:
                    self.event_catalogue = pickle.load(f)
                print(f"Loaded full catalogue from {full_catalogue_files[0]}: {len(self.event_catalogue)} events")
                return self.event_catalogue
            except Exception as e:
                print(f"Warning: Failed to load full catalogue: {e}")
                print("Attempting to load from per-file pickles...")
        
        # Load per-file pickles and merge
        per_file_pkl = [f for f in pkl_files if 'file_' in f and 'events.pkl' in f]
        for pkl_file in natsorted(per_file_pkl):
            pkl_path = os.path.join(pkl_dir, pkl_file)
            try:
                with open(pkl_path, 'rb') as f:
                    file_events = pickle.load(f)
                    self.event_catalogue.update(file_events)
            except Exception as e:
                print(f"Warning: Failed to load {pkl_file}: {e}")
        
        print(f"\nTotal events loaded: {len(self.event_catalogue)}")
        return self.event_catalogue

    def save_event_catalogue(self, output_pkl_dir: str = None, identifier: str = None):
        if not self.event_catalogue:
            print("Event catalogue is empty, nothing to save")
            return None
        
        if output_pkl_dir is None:
            output_pkl_dir = self._output_path
        
        if not os.path.exists(output_pkl_dir):
            os.makedirs(output_pkl_dir)
        
        if identifier is None:
            identifier = "event_catalogue"
        
        pkl_filename = f"{identifier}.pkl"
        pkl_path = os.path.join(output_pkl_dir, pkl_filename)
        
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.event_catalogue, f)
            print(f"Event catalogue saved to: {pkl_path}")
            print(f"Total events saved: {len(self.event_catalogue)}")
            return pkl_path
        except Exception as e:
            print(f"Error: Failed to save event catalogue to {pkl_path}: {e}")
            return None

    def remove_events_from_file(self, file_identifier, run: 'experiment.Run' = None):
        if not self.event_catalogue:
            print("Event catalogue is empty")
            return 0
        
        # Determine file index from identifier
        file_index = None
        filename = None
        
        if isinstance(file_identifier, int):
            # Direct file index provided
            file_index = file_identifier
            # Try to get filename if run provided
            if run and file_index < len(run.files):
                filename = os.path.basename(run.files[file_index].filepath)
        elif isinstance(file_identifier, str):
            # Filename provided - search for matching file index
            filename = os.path.basename(file_identifier)
            if run:
                for idx, datafile in enumerate(run.files):
                    if os.path.basename(datafile.filepath) == filename:
                        file_index = idx
                        break
            else:
                # Search all runs
                for r in self.runs:
                    for idx, datafile in enumerate(r.files):
                        if os.path.basename(datafile.filepath) == filename:
                            file_index = idx
                            break
                    if file_index is not None:
                        break
        
        if file_index is None:
            print(f"Could not locate file: {file_identifier}")
            return 0
        
        # Find and remove events from this file
        events_to_remove = [name for name, info in self.event_catalogue.items()
                           if info.get('file N') == file_index]
        
        num_removed = len(events_to_remove)
        for event_name in events_to_remove:
            del self.event_catalogue[event_name]
        
        if filename:
            print(f"Removed {num_removed} events from file {file_index}: {filename}")
        else:
            print(f"Removed {num_removed} events from file index {file_index}")
        
        return num_removed
    
    def remove_events_from_files(self, start_file_index: int, end_file_index: int):
        if not self.event_catalogue:
            print("Event catalogue is empty")
            return 0
        
        if start_file_index > end_file_index:
            print("Error: start_file_index must be <= end_file_index")
            return 0
        
        events_to_remove = [name for name, info in self.event_catalogue.items()
                           if start_file_index <= info.get('file N') <= end_file_index]
        
        num_removed = len(events_to_remove)
        for event_name in events_to_remove:
            del self.event_catalogue[event_name]
        
        print(f"Removed {num_removed} events from files {start_file_index} to {end_file_index}")
        
        return num_removed