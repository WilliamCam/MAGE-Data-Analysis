# Where you define rules for reading raw data files
import h5py

class DataIO:
    def __init__(self, parent, IQ_labels = ['-I', '-Q'], file_format = 'hdf5'):
        #parent must be of same class as file_heirarchy master
        self.parent = parent
        self.file_format = file_format
        self.iq_labels = IQ_labels
        # sets heirarchy of raw data file so data can be read must contain 'Detectors' 'Channels' 'Data' 'Attributes'
        # TODO: move to .json
        self.file_heirarchy ={'Detectors' : {'Channels' : 'Data'}}
        self.attributes_heirarchy = {'Detectors': 'Attributes'}
        self.group_heirarchy = {'Experiment' : {'Run' : 'File'}}
        if file_format == 'hdf5':
            self.read_data = self.read_hdf5_file
            self.read_attributes = self.read_hdf5_attributes
        else:
            raise ValueError("Requested file format method unrecognised")

    def read_hdf5_file(self, file):
        with h5py.File(file, 'r') as f:
            return hdf5_to_dict(f)
    def read_hdf5_attributes(self, h5_item):
            return dict(h5_item.attrs)
    
    def load_data(self, datafile, is_iq=True):
        if not self.parent:
            raise ValueError("Run has no parent Experiment")
        _exp = self.parent
        filepath = datafile.filepath
        _data, _attributes = self.read_data(filepath)
        datafile.metadata = _attributes
        # Load all channel data first
        for _detector_name in _data.keys():
            _detector = datafile.detectors[_detector_name]
            for _channel_name in _data[_detector_name].keys():
                #TODO error for IQ if data already loaded.
                try:
                    _channel = _detector.channels[_channel_name]
                    _channel.data = _data[_detector_name][_channel_name]['Data']*_exp.scaling_gain
                except:
                    if self.iq_labels[0] in _channel_name:
                        _channel = _detector.channels[_channel_name.strip(self.iq_labels[0])]
                        _channel.data['I'] = _data[_detector_name][_channel_name]['Data']*_exp.scaling_gain
                    if self.iq_labels[1] in _channel_name:
                        _channel = _detector.channels[_channel_name.strip(self.iq_labels[1])]
                        _channel.data['Q'] = _data[_detector_name][_channel_name]['Data']*_exp.scaling_gain
                    _channel.is_IQ = False
                
        
        # Consolidate I/Q pairs if requested
        if is_iq:
            self._consolidate_iq_channels(datafile)

    def _consolidate_iq_channels(self, datafile):
        """Consolidate I/Q channel pairs into single complex channels."""
        for _detector_name, _detector in datafile.detectors.items():
            # Group channels by base name (removing I/Q suffixes)
            channel_groups = {}
            
            for _channel_name, _channel in list(_detector.channels.items()):
                if not _channel.is_IQ:
                    continue
                    
                # Find base name by removing I/Q suffixes
                base_name = _channel_name
                for suffix in self.iq_labels:
                    base_name = base_name.replace(suffix, '')
                
                if base_name not in channel_groups:
                    channel_groups[base_name] = []
                channel_groups[base_name].append((_channel_name, _channel))
            
            # Process each group
            for base_name, channels in channel_groups.items():
                if len(channels) == 2:
                    # Found I/Q pair
                    ch1_name, ch1 = channels[0]
                    ch2_name, ch2 = channels[1]
                    
                    # Determine which is I and which is Q
                    if self.iq_labels[0] in ch1_name and self.iq_labels[1] in ch2_name:
                        i_channel, q_channel = ch1, ch2
                    elif self.iq_labels[1] in ch1_name and self.iq_labels[0] in ch2_name:
                        i_channel, q_channel = ch2, ch1
                    else:
                        continue  # Not a proper I/Q pair
                    
                    # Consolidate the pair
                    i_channel.consolidate_iq_pair(q_channel)
                    
                    # Remove the Q channel from detector
                    del _detector.channels[q_channel.name]
                    
                    # Rename the consolidated channel in the detector
                    del _detector.channels[ch1_name]
                    _detector.channels[base_name] = i_channel
            
 


def hdf5_to_dict(h5_item):
    """
    Recursively converts an HDF5 group or file into a Python dictionary.
    """
    data_dict = {}
    attribute_dict = {}
    for key in h5_item.keys():
        item = h5_item[key]
        if item.attrs:
            attribute_dict[key] = {}
            for attr_name in item.attrs.keys():
                if attr_name in attribute_dict[key]:
                    continue
                try: 
                    val = item.attrs[attr_name]
                    attribute_dict[key][attr_name] = val
                except ValueError as e:
                    #print(f"Skipping attribute '{attr_name}': {e}")
                    # Optional: store a placeholder string instead
                    attribute_dict[key][attr_name] = "Incompatible HDF5 Type"
        if isinstance(item, h5py.Group):
            data_dict[key], _ = hdf5_to_dict(item)

        
        elif isinstance(item, h5py.Dataset):
            data_dict[key] = item[()]
            
    return data_dict, attribute_dict







