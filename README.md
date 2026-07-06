# MAGE Data Analysis
This is a public repo designed to organise data analysis scripts and procedures for the Multimode Acoustic Gravitaional Wave Experiemt (MAGE).


# MAGE Data Analysis Package Tutorial

This guide documents the package in the src directory and is intended to act as a practical tutorial for loading experiment data, inspecting channels, fitting resonances, and searching for events.
It is based on the workflow shown in [examples/filter_search_example.ipynb](examples/filter_search_example.ipynb).

## 1. What this package does

The package is organized around a small analysis pipeline for detector data stored in HDF5 files:

- [experiment.py](experiment.py): defines the high-level Experiment container.
- [run.py](run.py): groups files into logical runs.
- [fileIO_rules.py](fileIO_rules.py): reads HDF5 data and builds the detector/channel structure.
- [channel.py](channel.py): represents detectors and channels, loads data, fits Lorentzian peaks, and supports simple processing.
- [crystal.py](crystal.py) and [SQUID.py](SQUID.py): hold calibration data loaded from YAML files.
- [OptimalFilter.py](OptimalFilter.py): implements the optimal-filter based event search workflow.

In practice, you usually:

1. Create an Experiment or FilterSearch object.
2. Point it at a directory containing run folders and HDF5 files.
3. Load the metadata and structure of the experiment.
4. Read calibration files.
5. Load a specific file and inspect a channel.
6. Fit a Lorentzian resonance for the channel.
7. Run the event-search workflow across files.

---

## 2. Dependencies

The package expects a Python environment with the following packages installed:

- numpy
- h5py
- scipy
- matplotlib
- lmfit
- pyyaml
- natsort

If you are working in a notebook like the example, the imports are already set up for you. In a standalone script, make sure the src directory is on the Python path.

```python
import os
import sys

sys.path.append(os.path.abspath("src"))
```

---

## 3. The basic workflow

The example notebook demonstrates the core sequence. A minimal version is shown below:

```python
import os
import sys

sys.path.append(os.path.abspath("src"))

from experiment import Experiment
from OptimalFilter import FilterSearch

path = os.path.abspath("test_data")
exp = FilterSearch("test", path, "src/examples/example_config.yaml")
```

This creates a FilterSearch object that will inspect the data in the test_data directory using the example configuration.

---

## 4. Understanding the main classes

### 4.1 Experiment

The Experiment object is the top-level container for a dataset. It is responsible for:

- storing the runs in the experiment,
- reading the directory structure and metadata,
- building the detector/channel tree,
- loading calibration data,
- providing helper methods for inspection.

Typical usage:

```python
exp = Experiment("my_experiment", "path/to/data", "path/to/config.yaml")
exp.show_tree(show_channels=True)
```

The show_tree() method prints a simple tree showing the hierarchy of runs, files, detectors, and channels.

### 4.2 Run

A Run groups multiple data files that belong to a single acquisition or data collection segment.

```python
run = exp.runs[0]
print(run.name)
print(len(run.files))
```

You can inspect the files in a run and find the channels available within them.

### 4.3 DataFile and DataIO

DataFile stores the raw file path and its loaded detector/channel objects. The DataIO class handles HDF5 reading.

When you load a file, the package will populate the detector/channel structure from the HDF5 contents.

```python
datafile = exp.runs[0].files[0]
exp.load_file(datafile)
```

### 4.4 Channel and Detector

A Detector is a container for channels. A Channel stores data and can be processed, fitted, and inspected.

You can access a channel like this:

```python
channel = exp.runs[0].files[0].detectors["AI 1"].channels["CH 5"]
```

The channel data may be loaded as a raw array or as an I/Q pair depending on the file contents.

---

## 5. Loading and inspecting data

### 5.1 Show the experiment tree

```python
exp.show_tree(show_channels=True)
```

This is useful for confirming that the directory layout and channel names match your expectations.

### 5.2 Load a file

```python
datafile = exp.runs[0].files[0]
exp.load_file(datafile)
```

This loads the HDF5 data into the corresponding Channel objects.

### 5.3 Access a channel

```python
channel = datafile.detectors["AI 0"].channels["CH 10"]
print(channel.name)
print(type(channel.data))
```

If the file contains I- and Q-channel data, the package may consolidate them into a single channel with I and Q data stored in a dictionary.

---

## 6. Calibration workflow

The package expects calibration data to be available through YAML files. The example configuration in [examples/example_config.yaml](examples/example_config.yaml) maps detectors to SQUID and crystal calibrations.

### 6.1 Read calibration data

```python
exp.read_calibration("src/calibration_files")
```

This will:

- load the detector-to-mode mapping from the config YAML,
- read the SQUID calibration YAML files,
- read the crystal calibration YAML files,
- populate exp.squids and exp.crystals.

You can inspect the loaded calibration like this:

```python
print(exp.crystals["AI 0"].calibration_data)
print(exp.squids["AI 0"].calibration_data)
```

### 6.2 Why calibration matters

The event-search workflow uses calibration values such as:

- SQUID gain,
- crystal mode parameters,
- resonance frequency,
- effective mass,
- and other mode-specific values.

If calibration data is missing or incorrect, the fitting and event processing results will be unreliable.

---

## 7. Fitting a resonance with a Lorentzian model

The Channel class includes a fit_lorentzian() method that computes a power spectrum and fits a Lorentzian thermal peak.

```python
channel = exp.runs[0].files[0].detectors["AI 1"].channels["CH 5"]
result = channel.fit_lorentzian(
    fs=238.4185791015625,
    fdemod=4993066,
    nfft=2**13,
    span=300,
    Plot=True,
)

print(result)
```

The fitted result includes values like:

- centre frequency,
- linewidth,
- amplitude,
- Q factor,
- and associated errors.

The fit_result is stored in the channel object and can be reused later.

---

## 8. Running the event-search pipeline

The main search workflow is implemented by FilterSearch, which inherits from Experiment and adds event detection.

```python
search = FilterSearch("test", path, "src/examples/example_config.yaml")
```

To search all files in a run:

```python
out = search.search_all_files(
    search.runs[0],
    show_plot=False,
    NFFT=2**13,
)
```

The search routine:

1. loads each file,
2. fits channel resonances,
3. calibrates the strain signal,
4. builds an optimal-filter template,
5. detects candidate events based on SNR,
6. saves event pickles for later inspection.

The returned event catalogue is stored in search.event_catalogue.

### 8.1 Inspecting detected events

```python
search.print_events()
```

If you want to inspect a specific event, use:

```python
search.inspect_event(
    search.runs[0],
    "270326-10:01:41-Det AI 0-chCH 10-SNR 4.29",
    span=500,
)
```

This plots or visualizes the event around the detected trigger time.

---

## 9. A complete example script

Here is a complete example that follows the notebook closely:

```python
import os
import sys

sys.path.append(os.path.abspath("src"))

from OptimalFilter import FilterSearch

base_path = os.path.abspath("test_data")
config_path = os.path.abspath("src/examples/example_config.yaml")

search = FilterSearch("test", base_path, config_path)
search.read_calibration("src/calibration_files")

search.show_tree(show_channels=True)

# Load the first file and inspect a channel
first_file = search.runs[0].files[0]
search.load_file(first_file)

channel = first_file.detectors["AI 1"].channels["CH 5"]
channel.fit_lorentzian(
    fs=238.4185791015625,
    fdemod=4993066,
    nfft=2**13,
    span=300,
    Plot=True,
)

# Run event search over the first run
search.search_all_files(search.runs[0], show_plot=False, NFFT=2**13)
search.print_events()
```

---

## 10. Notes about data representation

The package is built around a hierarchical model:

- Experiment contains Runs
- Run contains DataFiles
- DataFile contains Detectors
- Detector contains Channels

Each Channel can contain:

- raw data,
- fitted parameters,
- event metadata,
- and temporary processing results.

This structure makes it easy to navigate through the data even when the underlying HDF5 files are complex.

---

## 11. Common troubleshooting tips

- If imports fail, confirm that the src directory is on your Python path.
- If channels are missing, check the file layout and the detector/channel names in the HDF5 file.
- If calibration loading fails, verify that the YAML files exist and that the config file references them correctly.
- If fitting produces poor results, try different values for fs, fdemod, nfft, or span.
- If event detection returns too many or too few events, adjust the SNR threshold and the search parameters.

---

## 12. Recommended next steps

Once you are comfortable with the basic workflow, you can:

- inspect more channels across multiple files,
- compare the fit quality for different modes,
- adjust the search parameters for your own dataset,
- and extend the pipeline for additional analysis steps.

The notebook in [examples/filter_search_example.ipynb](examples/filter_search_example.ipynb) is the best place to start if you want a hands-on version of the same workflow.
