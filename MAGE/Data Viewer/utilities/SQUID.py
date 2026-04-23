#SQUID amplifier
import numpy as np
class SQUID:
    """Calibration data for a SQUID amplifier used in a detector"""
    def __init__(self, name: str, parent):
        self.name = name
        self.detector_name = None  # Will be set when added to Experiment
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