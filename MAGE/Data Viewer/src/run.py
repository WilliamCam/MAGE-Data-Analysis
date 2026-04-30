from typing import List, Dict, Callable, Optional
import numpy as np
from fileIO_rules import DataFile
import os

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

    def stitch_channels(self, detector_name: str, channel_name: str, file_list = []) -> Dict[str, np.ndarray]:
        """
        Stitch data from the same detector and channel across all files in the run.
        Returns a dictionary of concatenated numpy arrays for each data key (e.g., 'I', 'Q').
        
        Args:
            detector_name: Name of the detector
            channel_name: Name of the channel
            
        Returns:
            Dict with keys like 'I', 'Q' and values as concatenated numpy arrays
        """
        stitched_data = {}
        if not file_list:
            _files = self.files
        else:
            _files = file_list
        for datafile in _files:
            if detector_name in datafile.detectors and channel_name in datafile.detectors[detector_name].channels:
                channel = datafile.detectors[detector_name].channels[channel_name]
                if hasattr(channel, 'data') and isinstance(channel.data, dict):
                    for key, value in channel.data.items():
                        if key not in stitched_data:
                            stitched_data[key] = []
                        stitched_data[key].append(value)
        
        # Concatenate the arrays
        for key in stitched_data:
            stitched_data[key] = np.concatenate(stitched_data[key])
        
        return stitched_data