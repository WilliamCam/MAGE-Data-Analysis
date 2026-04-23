#Utilities for analysing transient events in MAGE data files.
import Analysis_functions
import dataStream
import numpy as np
import os

#Helper function to read calibration data.
def read_two_column_data(file_path):
    data = np.loadtxt(file_path)
    column_1 = data[:, 0]  # First column
    column_2 = data[:, 1]  # Second column
    return np.array([column_1, column_2])

def import_calibration_data(calibration_directory, run:dataStream.Run, config_yaml_path=None):
    """
    Load calibration data for a run from config.yaml and corresponding YAML files.
    
    Args:
        calibration_directory: Path to directory containing config.yaml and calibration files
        run: dataStream.Run object to populate with calibration data
        config_yaml_path: Optional path to config.yaml (defaults to calibration_directory/config.yaml)
    
    Returns:
        Tuple of (squids dict, crystals dict) from the parent Experiment
    """
    if config_yaml_path is None:
        config_yaml_path = os.path.join(calibration_directory, 'config.yaml')
    
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")
    
    # Use parent Experiment's read_calibration method
    if run.parent:
        experiment = run.parent
        return experiment.read_calibration(config_yaml_path)
    else:
        raise ValueError("Run object has no parent Experiment")




def optimal_filter_search(run:dataStream.Run, template, output_directory, 
        calibration_directory = None, co_analysis=False, show_plot=False, 
        Lin = 400e-9):
    files = run.files
    if calibration_directory:
        pass
    pass





















def large_events_by_SNR(event_catalogue, detectors, modes, SNR_bounds=[5.0,1000.0]):
    """
    Selects and saves large events with SNR greater than min_snr and smaller than max_snr,
    along with other event details (time, SNR, input AI, channel, frequency, amplitude, file N, index).
    
    Args:
        event_catalogue (dict): Dictionary containing event data, where each event has keys like 'SNR', 'input AI', 'channel', etc.
        min_snr (float): Minimum SNR value.
        max_snr (float): Maximum SNR value.
        
    Returns:
        dict: Dictionary with event channels and input AI values as keys, containing lists of large event details.
    """
    large_events = {}
    min_snr, max_snr = SNR_bounds
    # Loop over all unique values for 'input AI' and 'channel' and find large events based on SNR
    for ai_value, det_name in enumerate(detectors): 
        for channel, mode_name in enumerate(modes):
            # Get the list of events for a specific channel and input AI
            key = f"{det_name}_{mode_name}_events"
            large_event_details = []  # List to hold all details for large events

            # Loop over the event_catalogue and apply condition based on 'SNR'
            for index, event in enumerate(event_catalogue):
                if (event_catalogue[event]['input AI'] == ai_value and
                    event_catalogue[event]['channel'] == channel):
                    
                    snr_value = event_catalogue[event]['SNR']
                    if min_snr < snr_value < max_snr:  # Check if SNR is within the specified range
                        # Collect all relevant event information
                        event_info = {
                            'time': event_catalogue[event]['time'],
                            'SNR': snr_value,
                            'input AI': event_catalogue[event]['input AI'],
                            'channel': event_catalogue[event]['channel'],
                            'frequency': event_catalogue[event]['frequency'],
                            'amplitude': event_catalogue[event]['amplitude'],
                            'file N': event_catalogue[event]['file N'],
                            'index': event_catalogue[event]['index']
                        }
                        #### Skip bad files
                        #if event_catalogue[event]['file N'] in [243, 244, 245, 246, 247, 248]: # for run 10
                        #if event_catalogue[event]['file N'] in [69]: # for run 11
                            #continue
                        large_event_details.append(event_info)

            # Save the large event details
            large_events[key] = large_event_details

    return large_events