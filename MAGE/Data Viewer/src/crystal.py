import numpy as np
import yaml
class Crystal:
    """Quartz crystal calibration data"""
    def __init__(self, name: str, parent):
        self._name = name
        self.detector_name = None  # Will be set when added to Experiment
        self._parent = parent
        self.calibration_data = {}
    
    def crystal_config(self, config_yaml):
        with open(config_yaml, 'r') as f:
            crystal_data = yaml.safe_load(f)
            # Parse modes from YAML and create calibration_data dict
            for mode in crystal_data.get('modes', []):
                mode_name = mode['name']
                self.calibration_data[mode_name] = {key: value for key, value in mode.items() if key != 'name'}
            
                