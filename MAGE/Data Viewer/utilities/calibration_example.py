"""
Example usage of calibration loading functionality.

This script demonstrates how to:
1. Read calibration data from config.yaml
2. Load corresponding SQUID and Crystal YAML files
3. Access calibration information for specific channels and modes
"""

from dataStream import Experiment
import os

# Example usage
def main():
    # Path to the utilities directory where YAML config files are stored
    utilities_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(utilities_dir, 'config.yaml')
    
    # Create Experiment object (with read_metadata_on_init=False to skip HDF5 reading for this example)
    experiment = Experiment(
        name="MAGE_Example",
        master_filepath=utilities_dir,
        config_yaml=config_path,
        read_metadata_on_init=False  # Set to True if you have actual HDF5 files
    )
    
    # Load calibration data from config.yaml and corresponding YAML files
    squids, crystals = experiment.read_calibration(config_path)
    
    print(f"Loaded {len(squids)} SQUID configurations")
    print(f"Loaded {len(crystals)} Crystal configurations")
    print()
    
    # Example 1: Access all SQUID calibration data
    print("=== SQUID Calibration Data ===")
    for detector_name, squid in squids.items():
        print(f"\nDetector: {detector_name}")
        print(f"  SQUID Name: {squid.name}")
        print(f"  Calibration Data: {squid.calibration_data}")
    
    # Example 2: Access all Crystal modes for a detector
    print("\n=== Crystal Calibration Data ===")
    for detector_name, crystal in crystals.items():
        print(f"\nDetector: {detector_name}")
        print(f"  Crystal Name: {crystal._name}")
        print(f"  Modes: {list(crystal.calibration_data.keys())}")
        for mode_name, mode_data in crystal.calibration_data.items():
            print(f"    {mode_name}: Meff={mode_data['Meff']}, xi={mode_data['xi']}, Rlambda={mode_data['Rlambda']}")
    
    # Example 3: Get calibration for a specific channel
    print("\n=== Channel-Specific Calibration ===")
    detector = "AI 0"
    mode = "C300"
    calib = experiment.get_channel_calibration(detector, mode)
    print(f"Calibration for {detector}/{mode}:")
    print(f"  SQUID: {calib.get('SQUID')}")
    print(f"  Crystal: {calib.get('Crystal')}")


if __name__ == "__main__":
    main()
