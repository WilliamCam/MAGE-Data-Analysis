import h5py
import numpy as np
import os
import os
from SQUID import SQUID
from crystal import Crystal
from fileIO_rules import DataIO, DataFile
from channel import Channel, Detector
from run import Run
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
            self.read_metadata()
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
            
            # Build nested detector-channel structure
            detectors = {}
            detector_names = list(data_tree.keys())
            for det_name in detector_names:
                detectors[det_name] = {'Channels': list(data_tree[det_name].keys())}
            
            _temp_metadata[_run_dir]['Files'] = {file_name: {'Detectors': detectors} for file_name in _files}
            
            # Handle attributes if present
            if attributes_tree.get('Detectors') == "Attributes":
                _temp_metadata[_run_dir]["Attributes"] = {}
                for _data_key in detector_names:
                    _temp_metadata[_run_dir]["Attributes"][_data_key] = _attributes.get(_data_key, {})
        self.metadata = _temp_metadata            
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

            # Build nested detector-channel structure per file
            file_metadata = {}
            for datafile in run.files:
                file_name = os.path.basename(datafile.filepath)
                detectors = {}
                for det_name, det in datafile.detectors.items():
                    detectors[det_name] = {'Channels': list(det.channels.keys())}
                file_metadata[file_name] = {'Detectors': detectors}
            
            updated_metadata[run_name]['Files'] = file_metadata

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
            for _file_name in _meta_data['Files'].keys():
                _file_meta = _meta_data['Files'][_file_name]
                _absolute_path = os.path.join(self.exp_directory, _run, _file_name)
                _active_file = DataFile(_active_run, _absolute_path)
                _active_run.add_file(_active_file)
                for _detector_name, _detector_info in _file_meta['Detectors'].items():
                    _active_detector = Detector(_active_file, _detector_name)
                    _active_file.detectors[_detector_name] = _active_detector
                    for _channel in _detector_info['Channels']:
                        #init channel without any data
                        _active_channel = Channel(_active_detector, _channel)
                        _active_detector.channels[_channel] = _active_channel
    
    def load_file(self, datafile, update=True, is_IQ = True):
        self.dataIO.load_data(datafile)
        if update:
            self.update_metadata()

    def show_tree(self, show_channels=False):
        """
        Print an ASCII tree showing the hierarchical structure of the experiment.

        Args:
            show_channels: If True, extend the tree to show detectors, channels,
                and any loaded dataset names for each file.
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

                if not show_channels:
                    continue

                # Show detectors and channels loaded in this file
                detector_prefix = run_connector + ("    " if j == len(run.files) - 1 else "│   ")
                if not datafile.detectors:
                    continue

                for k, detector in enumerate(datafile.detectors.values()):
                    detector_connector = "└──" if k == len(datafile.detectors) - 1 else "├──"
                    print(f"{detector_prefix}{detector_connector} Detector: {detector.name}")

                    if not detector.channels:
                        continue

                    visible_channels = [ch for ch in detector.channels.values()
                                        if isinstance(ch.data, dict) and ch.data]
                    if not visible_channels:
                        continue

                    channel_prefix = detector_prefix + ("    " if k == len(datafile.detectors) - 1 else "│   ")
                    for l, channel in enumerate(visible_channels):
                        channel_connector = "└──" if l == len(visible_channels) - 1 else "├──"
                        print(f"{channel_prefix}{channel_connector} Channel: {channel.name}")

                        dataset_prefix = channel_prefix + ("    " if l == len(visible_channels) - 1 else "│   ")
                        for m, dataset_name in enumerate(channel.data.keys()):
                            dataset_connector = "└──" if m == len(channel.data) - 1 else "├──"
                            print(f"{dataset_prefix}{dataset_connector} Dataset: {dataset_name}")
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
                squid = SQUID(squid_name, self)
                squid.detector_name = detector_name  # Store detector mapping
                squid.squid_config(squid_file)
                self.add_squid(squid)
            else:
                print(f"Warning: SQUID file not found: {squid_file}")
            
            crystal_name = detector_config['crystal']['name']
            crystal_file = os.path.join(calibration_dir, f"{crystal_name}.yaml")
            
            if os.path.exists(crystal_file):
                # Create Crystal object and populate with YAML data
                crystal = Crystal(crystal_name, self)
                crystal.detector_name = detector_name  # Store detector mapping
                crystal.crystal_config(crystal_file)
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




            

            
             
    



