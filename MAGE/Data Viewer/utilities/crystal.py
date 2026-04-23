import numpy as np
class Crystal:
    """Quartz crystal calibration data"""
    def __init__(self, name: str, parent):
        self._name = name
        self.detector_name = None  # Will be set when added to Experiment
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
    
 