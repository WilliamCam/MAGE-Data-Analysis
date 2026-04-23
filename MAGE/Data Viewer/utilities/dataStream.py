import h5py
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from os import listdir
import gc
from scipy import signal
from scipy.signal import welch, csd
from scipy.signal import find_peaks
import lmfit
from scipy.fft import fft, fftfreq
import scipy.constants
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
import sys
import os
from typing import List, Dict, Callable, Optional
import fnmatch
from SQUID import SQUID
from crystal import Crystal
from fileIO_rules import DataIO
from natsort import natsorted
import yaml


# ...existing code...

from typing import Optional, List, Dict, Callable
import re

class Experiment:
    """Container for multiple Runs (each run contains multiple DataFile instances)."""
    def __init__(self, name: str, master_filepath, config_yaml, read_metadata_on_init = True, **kwargs):
        self.name = name
        self._config = config_yaml
        self.config: Dict[str, Dict[str, str]] = {}  # Mapping detector -> {channel -> mode_name}
        self.runs: List[Run] = []
        self.exp_directory = master_filepath
        self._output_path = kwargs.pop('output_path', master_filepath)
        self.detector_names = {}
        self.channel_names = {}
        self.squids: Dict[str, SQUID] = {}  # Mapping of detector name to SQUID calibration data
        self.crystals: Dict[str, Crystal] = {}  # Mapping of detector name to Crystal calibration data
        self.dataIO = DataIO(self)
        
        self.scaling_gain = kwargs.pop('scaling_gain',3.14e-5)
        if read_metadata_on_init:
            print(self.exp_directory)
            self.metadata = self.read_metadata()
            self._populate_structures()
            self.show_tree()
                
    def add_run(self, run):
        self.runs.append(run)
    
    def iter_runs(self):
        return iter(self.runs)
    
    def add_squid(self, squid: SQUID):
        """Add SQUID calibration data for a detector."""
        self.squids[squid.detector_name] = squid

    def add_crystal(self, crystal: Crystal):
        """Add crystal calibration data for a detector."""
        self.crystals[crystal.detector_name] = crystal

    def __len__(self):
        return len(self.runs)  
    
    def read_metadata(self, file_index = 0, attributes=True):
        filepath = self.exp_directory
        _temp_metadata = {}
        heirarchy = (self.dataIO.file_heirarchy, self.dataIO.attributes_heirarchy)
        #Looks at local data stored in experiment directory to read in valuable information, channel names, number of files etc..
        #Data must be stored by run, files with no added files other than raw data
        for _run_dir in os.listdir(filepath):
            if _run_dir == '.DS_Store':
                continue
            _run_path = os.path.join(filepath, _run_dir)
            _files = natsorted(os.listdir(_run_path))
            #print(_run_path)
            meta_file = _files[file_index]
            #populate run names
            _temp_metadata[_run_dir] = {}
            _temp_metadata[_run_dir]['N_files'] = len(_files)
            _temp_metadata[_run_dir]['Files'] = list(_files)
            _temp_data, _attributes = self.dataIO.read_data(os.path.join(_run_path,meta_file))
            heirarchy_tree, attributes_tree = heirarchy
            data_tree = _temp_data
            while isinstance(heirarchy_tree, dict) and len(heirarchy_tree) == 1:
                h_key = next(iter(heirarchy_tree))
                data_keys = list(data_tree.keys())
                _temp_metadata[_run_dir][h_key] = data_keys
                data_tree = data_tree[data_keys[0]]
                heirarchy_tree = heirarchy_tree[h_key]
                if attributes_tree.get(h_key) == "Attributes":
                    _temp_metadata[_run_dir]["Attributes"] = {}
                    for _data_key in data_keys:
                        _temp_metadata[_run_dir]["Attributes"][_data_key] = _attributes[_data_key]
        return _temp_metadata

    def update_metadata(self):
        """
        Update metadata dictionary based on current experiment structures.
        Traverses the populated runs, files, detectors, and channels to rebuild
        the metadata dictionary that reflects the current state.
        """
        updated_metadata = {}

        for run in self.runs:
            run_name = run.name
            updated_metadata[run_name] = {}

            # Count files in this run
            updated_metadata[run_name]['N_files'] = len(run.files)
            updated_metadata[run_name]['Files'] = [os.path.basename(f.filepath) for f in run.files]

            # Collect unique detectors across all files in this run
            detectors_in_run = set()
            channels_in_run = set()

            for datafile in run.files:
                detectors_in_run.update(datafile.detectors.keys())
                for detector in datafile.detectors.values():
                    channels_in_run.update(detector.channels.keys())

            updated_metadata[run_name]['Detectors'] = list(detectors_in_run)
            updated_metadata[run_name]['Channels'] = list(channels_in_run)

            # Initialize attributes if they exist in original metadata
            if hasattr(self, 'metadata') and run_name in self.metadata:
                if 'Attributes' in self.metadata[run_name]:
                    updated_metadata[run_name]['Attributes'] = self.metadata[run_name]['Attributes']
                else:
                    updated_metadata[run_name]['Attributes'] = {}

        # Update the instance metadata
        self.metadata = updated_metadata

    def _populate_structures(self, meta_dict = {}):
        #uses metadata to initialise all data structures in Experiment
        #Doesnt actually read in any data just populates the framework
        if not meta_dict:
            _meta_data = self.metadata
        else:
            _meta_data = meta_dict
        for _run in _meta_data.keys():
            _active_run = Run(self, _run, filepath=os.path.join(self.exp_directory, _run))
            self.add_run(_active_run)
            _meta_data = _meta_data[_run]
            for _file in _meta_data['Files']:
                _absolute_path = os.path.join(self.exp_directory, _run, _file)
                _active_file = DataFile(_active_run, _absolute_path)
                _active_run.add_file(_active_file)
                for _detector in _meta_data['Detectors']:
                    _active_detector = Detector(_active_file, _detector)
                    _active_file.detectors[_detector] = _active_detector
                    for _channel in _meta_data['Channels']:
                        #init channel without any data
                        _active_channel = Channel(_active_detector, _channel)
                        _active_detector.channels[_channel] = _active_channel
    
    def load_file(self, datafile, update=True, is_IQ = True):
        self.dataIO.load_data(datafile)
        if update:
            self.update_metadata()

    def show_tree(self):
        """
        Print an ASCII tree showing the hierarchical structure of the experiment.
        """
        print(f"Experiment: {self.name}")
        
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

    def read_calibration(self, calibration_dir):
        """
        Read calibration data from config.yaml and load corresponding SQUID and Crystal YAML files.
        Populates self.squids, self.crystals, and self.config dictionaries.
        
        Args:
            calibration_dir: Directory containing config.yaml and calibration YAML files
        """
        
        # Read config.yaml
        with open(self._config, 'r') as f:
            config = yaml.safe_load(f)
   
        # Process each detector in the config
        for detector_config in config.get('Detectors', []):
            detector_name = detector_config['name']
            
            # Store channel-to-mode mapping for this detector
            self.config[detector_name] = {}
            for mode_mapping in detector_config.get('modes', []):
                channel_name = mode_mapping['channel']
                mode_name = mode_mapping['name']
                self.config[detector_name][channel_name] = mode_name
            
            # Load SQUID calibration data
            squid_name = detector_config['SQUID']['name']
            squid_file = os.path.join(calibration_dir, f"{squid_name}.yaml")
            
            if os.path.exists(squid_file):
                with open(squid_file, 'r') as f:
                    squid_data = yaml.safe_load(f)
                
                # Create SQUID object and populate with YAML data
                squid = SQUID(squid_name, self)
                squid.detector_name = detector_name  # Store detector mapping
                squid.calibration_data = {
                    'Vphi': squid_data.get('Vphi'),
                    'corner_frequency': squid_data.get('corner_frequency'),
                    'Min': squid_data.get('Min'),
                    'serial_number': squid_data.get('serial_number')
                }
                self.add_squid(squid)
            else:
                print(f"Warning: SQUID file not found: {squid_file}")
            
            # Load Crystal calibration data
            crystal_name = detector_config['crystal']['name']
            crystal_file = os.path.join(calibration_dir, f"{crystal_name}.yaml")
            
            if os.path.exists(crystal_file):
                with open(crystal_file, 'r') as f:
                    crystal_data = yaml.safe_load(f)
                
                # Create Crystal object and populate with YAML data
                crystal = Crystal(crystal_name, self)
                crystal.detector_name = detector_name  # Store detector mapping
                
                # Parse modes from YAML and create calibration_data dict
                for mode in crystal_data.get('modes', []):
                    mode_name = mode['name']
                    crystal.calibration_data[mode_name] = {
                        'Meff': mode.get('Meff'),
                        'xi': mode.get('xi'),
                        'Qi': mode.get('Qi'),
                        'Rlambda': mode.get('Rlambda')
                    }
                
                self.add_crystal(crystal)
            else:
                print(f"Warning: Crystal file not found: {crystal_file}")
        
        return self.squids, self.crystals

    def get_channel_calibration(self, detector_name: str, mode_name: str) -> Dict:
        """
        Retrieve calibration data for a specific channel/mode.
        """
        calibration = {}
        if detector_name in self.squids:
            squid = self.squids[detector_name]
            calibration['SQUID'] = squid.calibration_data
        
        if detector_name in self.crystals:
            crystal = self.crystals[detector_name]
            if mode_name in crystal.calibration_data:
                calibration['Crystal'] = crystal.calibration_data[mode_name]
        
        return calibration

    def get_mode_name_from_channel(self, detector_name: str, channel_name: str) -> str:
        """
        Get the mode name for a specific channel in a detector.
        
        Args:
            detector_name: Detector identifier (e.g., 'AI 0')
            channel_name: Channel identifier (e.g., 'CH 1')
            
        Returns:
            Mode name (e.g., 'C300') or None if mapping not found
        """
        if detector_name in self.config:
            return self.config[detector_name].get(channel_name)
        return None

    def get_channels_for_detector(self, detector_name: str) -> Dict[str, str]:
        """
        Get the channel-to-mode mapping for a specific detector.
        
        Args:
            detector_name: Detector identifier (e.g., 'AI 0')
            
        Returns:
            Dictionary mapping channel names to mode names (e.g., {'CH 1': 'C300', 'CH 2': 'C302', ...})
        """
        return self.config.get(detector_name, {})

    def get_all_channel_mode_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the complete channel-to-mode mapping for all detectors.
        
        Returns:
            Dictionary mapping detector names to channel-to-mode dictionaries
        """
        return self.config

class Run:
    """Container for multiple DataFile instances that belong to a single run."""
    def __init__(self, parent, name: str, filepath: Optional[str] = None, **kwargs):
        self.name = name
        self.parent = parent
        _scaling_gain = kwargs.pop('scaling_gain', 3.14e-5)
        #unique detector and channel names
        self.detector_names = {}
        self.channel_names = {}
        self.filepath = filepath
        if not self.filepath:
            if not self.parent:
                raise ValueError("No parent Experiment in Run")
            self.filepath = os.join(self.parent.filepath, self.name)
        self.files: List[DataFile] = []
        self.calibration_data = {}
        self.calibration_data['scaling_gain'] = _scaling_gain

    def add_file(self, datafile):
        self.files.append(datafile)
              
    def iter_files(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def find_channels(self, predicate: Callable) -> List:
        """Return list of Channel objects across all files matching predicate(channel)."""
        out = []
        for df in self.files:
            for det in df.detectors.values():
                for ch in det.channels.values():
                    if predicate(ch):
                        out.append(ch)
        return out

    def group_by_detector(self) -> Dict[str, List]:
        """Return mapping detector_name -> list of Channel across all files in this run."""
        groups: Dict[str, List] = {}
        for df in self.files:
            for det_name, det in df.detectors.items():
                groups.setdefault(det_name, []).extend(list(det.channels.values()))
        return groups
    
    def detectors_in_run(self) -> List[str]:
        """Return list of unique detector names across all files in this run."""
        _detectors = set()
        for df in self.files:
            _detectors.update(df.detectors.keys())
        self.detector_names = list(_detectors)
        return self.detector_names
    
    def channels_in_run(self) -> List[str]:
        """Return list of unique channel names across all files in this run."""
        _channels = set()
        for df in self.files:
            for det in df.detectors.values():
                _channels.update(det.channels.keys())
        self.channel_names = list(_channels)
        return self.channel_names

class DataFile():
    """A DataFile contains the raw data."""
    def __init__(self, parent, filepath: str, **kwargs):
        self.parent = parent
        self.filepath = filepath
        self.detectors = {}
        self.metadata = {}
        

class Detector():
    """A detector contains multiple channels."""
    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name
        self.channels = {}

    def __repr__(self):
        return f"<Detector {self.name} with {len(self.channels)} channels>"
    
    def add_channel(self, channel):
        self.channels[channel.name] = channel

    def get_channel(self, name: str):
        return self.channels.get(name)
    

    
    # Nice Pythonic access: det["ch0"]
    def __getitem__(self, key):
        return self.channels[key]

class Channel():
    # ...existing code...
    def __init__(self, parent, name, data = None, is_IQ = True, **kwargs):
        self.parent = parent
        self.data = data
        self.is_IQ = is_IQ
        self.IQ_identifier = kwargs.pop('IQ_identifier', ('-I', '-Q'))
        self.name = name
        self.fit_result = {}
        self.events = {}

    def apply(self, func, **kwargs):
        """Apply a processing function to the channel."""
        self.data = func(self.data, **kwargs)

    def clear_data(self):
        """Remove the channel's loaded datasets from memory."""
        self.data = None
        self.fit_result = {}
        gc.collect()

    def consolidate_iq_pair(self, iq_channel):
        """Consolidate this channel with an I/Q pair channel.
        
        Args:
            iq_channel: The paired I/Q channel to consolidate with
        """
        if not self.is_IQ or not iq_channel.is_IQ:
            return
            
        # Store I and Q data in a dictionary
        if self.IQ_identifier[0] in self.name and self.IQ_identifier[1] in iq_channel.name:
            # This is I channel, iq_channel is Q channel
            self.data = {
                'I': self.data,
                'Q': iq_channel.data
            }
        elif self.IQ_identifier[1] in self.name and self.IQ_identifier[0] in iq_channel.name:
            # This is Q channel, iq_channel is I channel
            self.data = {
                'I': iq_channel.data,
                'Q': self.data
            }
        
        # Update channel name to remove I/Q suffix
        base_name = self.name
        for suffix in self.IQ_identifier:
            base_name = base_name.replace(suffix, '')
        self.name = base_name
        self.is_IQ = False  # No longer an I/Q channel

    def fit_lorentzian(self, fs, nfft, show_plot=True, f_demod=0.0):
        # Determine PSD of channel data and fit lorentzian to mode thermal peak
        if not self.data:
            raise ValueError("No data initilised in Channel ${self.name}")
        if not self.is_IQ:
            dataI = self.data['I']
            dataQ = self.data['Q']
            fn, SdataI = welch(dataI, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fn, SdataQ = welch(dataQ, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fit_result_I = lorentzian_fit_thermalpeak(SdataI, fn, f_demod,  Plot=show_plot, span=300)
            fit_result_Q = lorentzian_fit_thermalpeak(SdataQ, fn, f_demod,  Plot=show_plot, span=300)
            self.fit_result['I'] = fit_result_I
            self.fit_result['Q'] = fit_result_Q
            return fit_result_I, fit_result_Q
        else:
            data = self.data
            fn, Sdata = welch(data, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fit_result = lorentzian_fit_thermalpeak(Sdata, fn, f_demod,  Plot=show_plot, span=300)
            self.fit_result = fit_result
            return fit_result
        
def lorentzian_fit_thermalpeak(mag, fn, fdemod, Plot=False, span=300):
    """
    Fit a Lorentzian model to thermal peak data and return the fitted parameters, their errors, and the Q-factor.

    Parameters:
    - mag: The magnitude data (spectrum).
    - fn: The frequency vector corresponding to the magnitude data.
    - fdemod: The demodulation frequency to adjust the resonance frequency.
    - Plot: Whether to plot the fit (default is False).
    - start: Start index for the fitting range (default is 0).
    - stop: Stop index for the fitting range (default is 1600).
    
    Returns:
    - f_res: The fitted resonance frequency (Hz).
    - sigma: The fitted Lorentzian width (Hz).
    - integral: The amplitude of the Lorentzian peak.
    - Q: The Q-factor of the resonance.
    - const: The constant offset in the data (from the linear background).
    - f_res_err: The error on the resonance frequency.
    - sigma_err: The error on the Lorentzian width.
    - Q_err: The error on the Q-factor.
    - const_err: The error on the constant background.
    """
    # Lorentzian and constant background models
    lor_mod = LorentzianModel(prefix='lor_')
    lin_mod = ConstantModel(prefix='lin_')
    peak_index = np.where(mag==np.max(mag))[0][0]
    if peak_index-span < 1:
        fn_fit = fn[0:peak_index+span]
        linear_mag = mag[0:peak_index+span]
    else:
        fn_fit = fn[peak_index-span:peak_index+span]
        linear_mag = mag[peak_index-span:peak_index+span]

    # Make initial guesses for the parameters
    pars = lor_mod.guess(linear_mag, x=fn_fit)
    pars.update(lin_mod.make_params())

    # Add the Lorentzian and constant models together
    mod = lin_mod + lor_mod

    # Perform the fit
    out = mod.fit(linear_mag, pars, x=fn_fit)

    # Extract the fitted parameters
    Gamma = out.params["lor_sigma"].value * 2  # Lorentzian width (FWHM), multiply by 2
    f_res = out.params["lor_center"].value + fdemod  # Resonance frequency, adjusted by demodulation frequency
    Q = f_res / Gamma  # Quality factor
    integral = out.params["lor_amplitude"].value  # Amplitude of the Lorentzian peak
    height = out.params["lor_height"].value #Unnormalised height of Lorentzian peak
    sigma = out.params["lor_sigma"].value  # Lorentzian width (standard deviation)
    noise_val = out.params["lin_c"].value

    # Calculate the errors (standard errors of the parameters)
    f_res_err = out.params["lor_center"].stderr if out.params["lor_center"].stderr else 0.99  # Error on f_res
    sigma_err = out.params["lor_sigma"].stderr if out.params["lor_sigma"].stderr else 0.99  # Error on sigma
    Q_err = Q * np.sqrt((f_res_err / f_res) ** 2 + (sigma_err / Gamma) ** 2)  # Error on Q


    # Plot the results if requested
    if Plot:
        plt.ion()
        fig = plt.figure("IMPA DOWNLOAD")
        plt.axis('tight')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(fn_fit, linear_mag, 'o', markersize=0.2)
        ax.set_title(fdemod)
        ax.plot(fn_fit, out.best_fit, '-', label='fit')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.legend()
        plt.pause(0.05)
        plt.draw()

    # Return the fitted parameters and their errors
    ret = {'centre_freqeuncy': f_res, 'linewidth': sigma, 'amplitude': integral, 'Q_factor' : Q, 'center_frequency_error': f_res_err, 'linewidth_error': sigma_err, 'Q_factor_error': Q_err, 'height': height, 'noise_level': noise_val}
    return ret



            

            
             
    



