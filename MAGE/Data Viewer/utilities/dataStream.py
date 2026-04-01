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

        # Rs, Meff, xi are used for calibration of units. To be passed by calibration operation
        self.mode_impedance = None
        self.mode_effective_mass=None
        self.mode_geom_coupling=None
        
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
        # Lin, Min, and Vphi used for SQUID calibration
        self.input_L = None
        self.input_M = None
        self.Vphi = None

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
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.detectors = {}
        self.metadata = {}
        #TODO:Load calibration data
        self.calibration_data = {}
        self.calibration_data['scaling_gain'] = 3.09758E-5

    def load_file(self, update_metadata = True, calibrate=True):
        """Load the HDF5 file and populate detectors and channels."""
        with h5py.File(self.filepath, 'r') as f:
            if update_metadata:
                self.metadata = Analysis_functions.get_meta_data(self.filepath)
            for det_name in f.keys():
                detector = Detector(det_name)
                det_group = f[det_name]
                _data={}
                for mode in det_group.keys():
                    mode_group = det_group[mode]
                    #TODO: Handle other IQ configurations
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
                                else:
                                    _data[_mode][_iq_label] = mode_group[dset][:]
                                if 'I' in _data[_mode] and 'Q' in _data[_mode]:
                                    channel = Channel(parent=self, detector=det_name, mode=_mode, data=_data[_mode], is_IQ=True)
                                    detector.add_channel(channel)
                self.detectors[det_name] = detector

    def get_detector(self, name: str):
        return self.detectors.get(name)
    
    # Nice Pythonic access: df["detector1"]
    def __getitem__(self, key):
        return self.detectors[key]

from typing import List, Dict, Callable, Optional
import fnmatch

class Run:
    """Container for multiple DataFile instances that belong to a single run."""
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.files: List[DataFile] = []

    def add_file(self, datafile: DataFile):
        self.files.append(datafile)

    def load_files_from_folder(self, folder: str, pattern: Optional[str] = None,
                               update_metadata: bool = True, calibrate: bool = True):
        """
        Load all .h5/.hdf5 files in folder (optional filename pattern) into this Run.
        """
        for fname in listdir(folder):
            if not fname.lower().endswith(('.h5', '.hdf5')):
                continue
            if pattern and not fnmatch.fnmatch(fname, pattern):
                continue
            path = os.path.join(folder, fname)
            df = DataFile(path)
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
    
# ...existing code...

class SQUID:
    """
    Hold SQUID params (Lin, Min, optional Vphi) per detector and apply them
    to Detector, DataFile or Run objects by matching detector.name.
    """
    def __init__(self):
        self.map: Dict[str, Dict[str, float]] = {}

    def load_from_file(self, path: str):
        """Parse lines like: DETNAME 1.23e-6 4.56e-9 [0.1] (whitespace or comma separated)."""
        self.map.clear()
        with open(path, 'r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p for p in re.split(r'[,\s]+', line) if p]
                if len(parts) < 3:
                    continue
                name = parts[0]
                try:
                    Lin = float(parts[1])
                    Min = float(parts[2])
                    Vphi = float(parts[3]) if len(parts) >= 4 else None
                except Exception:
                    continue
                entry = {'Lin': Lin, 'Min': Min}
                if Vphi is not None:
                    entry['Vphi'] = Vphi
                self.map[name] = entry

    def get_params_for_detector(self, detector_name: str) -> Optional[Dict[str, float]]:
        return self.map.get(detector_name)

    def apply_to_detector(self, detector: Detector) -> bool:
        """
        Apply params to a single Detector instance. Returns True if applied,
        False if no params found for that detector.
        """
        params = self.get_params_for_detector(detector.name)
        if not params:
            return False
        if 'Lin' in params:
            detector.input_L = params['Lin']
        if 'Min' in params:
            detector.input_M = params['Min']
        if 'Vphi' in params:
            detector.Vphi = params['Vphi']
        # propagate to channels for convenience
        for ch in detector.channels.values():
            setattr(ch, 'input_L', detector.input_L)
            setattr(ch, 'input_M', detector.input_M)
            setattr(ch, 'Vphi', detector.Vphi)
        return True

    def apply_to_file(self, datafile: DataFile, ignore_missing: bool = True) -> Dict[str, bool]:
        """
        Apply SQUID params to all detectors in a DataFile.
        Returns a mapping detector_name -> applied_bool.
        If ignore_missing is False, missing params will raise KeyError.
        """
        results: Dict[str, bool] = {}
        for det_name, det in datafile.detectors.items():
            applied = self.apply_to_detector(det)
            results[det_name] = applied
            if not applied and not ignore_missing:
                raise KeyError(f"No SQUID params for detector '{det_name}'")
        return results

    def apply_to_run(self, run: Run, ignore_missing: bool = True) -> Dict[str, Dict[str, bool]]:
        """
        Apply SQUID params to every DataFile in a Run.
        Returns mapping file_path -> {detector_name: applied_bool}.
        """
        summary: Dict[str, Dict[str, bool]] = {}
        for df in run.files:
            res = self.apply_to_file(df, ignore_missing=ignore_missing)
            summary[df.filepath] = res
        return summary
# ...existing code...
   