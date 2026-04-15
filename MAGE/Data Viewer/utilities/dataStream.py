import h5py
import numpy as np
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
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
import sys
import os
import Analysis_functions


# ...existing code...

from typing import Optional, List, Dict, Callable
import re

class Channel():
    # ...existing code...
    def __init__(self, parent, detector, mode, data, is_IQ = True, **kwargs):
        self.parent = parent
        self.detector = detector
        self.mode = mode
        self.data = data
        self.dataset_path = None  # dataset path inside HDF5
        self.starttime = None
        
        self.name = f"{mode}"
        #TDO: Handle non-IQ data
        if is_IQ:
            if 'I' not in self.data and 'Q' not in self.data:
                raise ValueError(f"Expected IQ data for channel {self.name}, but 'I' or 'Q' keys not found.")
    def __repr__(self):
        return f"Dataset:, detector={self.detector}, mode={self.mode})"

    
    def apply(self, func, **kwargs):
        """Apply a processing function to the channel."""
        self.data = func(self.data, **kwargs)

class Detector():
    """A detector contains multiple channels."""
    def __init__(self, name: str):
        self.name = name
        self.channels = {}

    def __repr__(self):
        return f"<Detector {self.name} with {len(self.channels)} channels>"
    
    def add_channel(self, channel: Channel):
        self.channels[channel.name] = channel

    def get_channel(self, name: str):
        return self.channels.get(name)
    
    # Nice Pythonic access: det["ch0"]
    def __getitem__(self, key):
        return self.channels[key]
    
class DataFile():
    """A DataFile contains multiple detectors."""
    def __init__(self, parent, filepath: str, **kwargs):
        _scaling_gain = kwargs.pop('scaling_gain', 3.14e-5)
        self.parent = parent
        self.filepath = filepath
        self.detectors = {}
        #TODO:Load calibration data
        self.calibration_data = {}
        self.calibration_data['scaling_gain'] = _scaling_gain
        if hasattr(self.parent, '_filepath'):
            if self.parent._filepath is None:
                self.parent._filepath = os.path.dirname(filepath)
                self.parent._output_path = os.path.dirname(self.parent._filepath) + '/' + self.parent.name + '_analysis_output'
        
    def load_file(self, update_metadata = True, calibrate=True):
        """Load the HDF5 file and populate detectors and channels."""
        with h5py.File(self.filepath, 'r') as f:
            for det_name in f.keys():
                detector = Detector(det_name)
                det_group = f[det_name]
                _data={}
                for mode in det_group.keys():
                    mode_group = det_group[mode]
                    #TODO: Handle other IQ configurations / naming conventions
                    for iq_index, iq_label in enumerate(['-I', '-Q']):
                        if iq_label in mode:
                            _mode = mode.replace(iq_label, '')
                            if _mode not in _data:
                                _data[_mode] = {}
                            _iq_label = iq_label.replace('-', '')
                            for dset in mode_group.keys():
                                if len(mode_group.keys()) != 1:
                                    raise ValueError(
                                        f"Expected exactly 1 dataset under '{det_name}/{_mode}/{iq_label}' "
                                        f"in file '{self.filepath}', found {len(mode_group.keys())}: {mode_group.keys()}"
                                    )
                                if calibrate:
                                    _data[_mode][_iq_label] = mode_group[dset][:] * self.calibration_data['scaling_gain']
                                    if self.parent.instrument_calibration_data.get('scaling_gain') == None:
                                        self.parent.instrument_calibration_data['scaling_gain'] = self.calibration_data['scaling_gain']
                                else:
                                    _data[_mode][_iq_label] = mode_group[dset][:]
                                if 'I' in _data[_mode] and 'Q' in _data[_mode]:
                                    channel = Channel(parent=self, detector=det_name, mode=_mode, data=_data[_mode], is_IQ=True)
                                    detector.add_channel(channel)
                self.detectors[det_name] = detector
        if update_metadata:
            self.parent.metadata = Analysis_functions.get_meta_data(self.filepath)
            self.parent.add_file(self)
    def get_detector(self, name: str):
        return self.detectors.get(name)
    
    # Nice Pythonic access: df["detector1"]
    def __getitem__(self, key):
        return self.detectors[key]

from typing import List, Dict, Callable, Optional
import fnmatch

class SQUID:
    """Calibration data for a SQUID amplifier used in a detector"""
    def __init__(self, name: str, parent: Detector):
        self.name = name
        self._parent = parent
        self.calibration_data = {}
        #TODO add optional Vphi vs frequency curve.

    def read_SQUID_data(self, file_path):
        """Read SQUID calibration data from a text file."""
        #file format is CSV containing rows: name, Vphi, fc, input_L, input_M
        data = np.genfromtxt(file_path, format='str', delimiter=',')
        for row in data:
            if row[0] == self.name:
                Vphi = float(row[1])
                fc = float(row[2])
                input_L = float(row[3])
                input_M = float(row[4])
                self.calibration_data = {'Vphi': Vphi, 'fc': fc, 'input_L': input_L, 'input_M': input_M}
            else:
                raise ValueError(f"SQUID name '{self.name}' not found in calibration file '{file_path}'")
        return self.calibration_data
            
class Crystal:
    """Quartz crystal calibration data"""
    def __init__(self, name: str, parent: Detector):
        self._name = name
        self._parent = parent
        self.calibration_data = {}
    def read_crystal_data(self, file_path):
        """Read crystal calibration data from a text file."""
        # file format is mode_name, Meff, Rs, xi
        data = np.genfromtxt(file_path, format='str', delimiter=',')
        for row in data:
            mode_name = row[0]
            Meff = float(row[1])
            Rs = float(row[2])
            xi = float(row[3])
            self.calibration_data[mode_name] = {'Meff': Meff, 'Rs': Rs, 'xi': xi}
        return self.calibration_data  
 
class Run:
    """Container for multiple DataFile instances that belong to a single run."""
    def __init__(self, parent, name: str, filepath: Optional[str] = None, **kwargs):
        self.name = name
        self.parent = parent
        self.metadata = {}  # Cache for metadata across all files in this run
        self._filepath = filepath
        self._output_path = kwargs.pop('output_path', os.path.join(filepath, 'analysis_output') if filepath else None)
        self.files: List[DataFile] = []
        self.detector_names = None  # Cache for detectors across all files
        self.channel_names = None  # Cache for channels across all files

        self.squids: Dict[str, SQUID] = {}  # Mapping of detector name to SQUID calibration data
        self.crystals: Dict[str, Crystal] = {}  # Mapping of detector name to Crystal calibration data

    def add_file(self, datafile: DataFile):
        self.files.append(datafile)

    def add_squid(self, squid: SQUID):
        """Add SQUID calibration data for a detector."""
        self.squids[squid.detector_name] = squid

    def add_crystal(self, crystal: Crystal):
        """Add crystal calibration data for a detector."""
        self.crystals[crystal.detector_name] = crystal
    
    
            

    def load_files_from_folder(self, folder: str, pattern: Optional[str] = None,
                               update_metadata: bool = True, calibrate: bool = True):
        """
        Load all .h5/.hdf5 files in folder (optional filename pattern) into this Run.
        """
        self._filepath = folder
        for fname in listdir(folder):
            if not fname.lower().endswith(('.h5', '.hdf5')):
                continue
            if pattern and not fnmatch.fnmatch(fname, pattern):
                continue
            path = os.path.join(folder, fname)
            df = DataFile(self, path)
            df.load_file(update_metadata=update_metadata, calibrate=calibrate)
            self.add_file(df)

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



            

            
             
    



