#SQUID amplifier
import numpy as np
import yaml
class SQUID:
    """Calibration data for a SQUID amplifier used in a detector"""
    def __init__(self, name: str, parent):
        self.name = name
        self.detector_name = None  # Will be set when added to Experiment
        self._parent = parent
        self.calibration_data = {}
        #TODO add optional Vphi vs frequency curve.
    def squid_config(self, config_yaml):
        with open(config_yaml, 'r') as f:
            squid_data = yaml.safe_load(f)
            self.calibration_data = {key: value for key, value in squid_data.items() if key != 'name'}
    def squid_gain(self):
        Vphi = self.calibration_data['Vphi']
        Min = self.calibration_data['Min']
        amp_gain = self.calibration_data['amplifier_gain']
        return Vphi*Min*amp_gain