#TODO: Separate codebase into modules, define different experiment types
from importlib.resources import path
import dataStream
import EventSearchUtils
from typing import Optional, List, Dict, Callable
import re
import pickle

class Experiment:
    """Container for multiple Runs (each run contains multiple DataFile instances)."""
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.runs: List[dataStream.Run] = []

    def add_run(self, run: dataStream.Run):
        self.runs.append(run)


    #TODO Load runs from folder structure

    def iter_runs(self):
        return iter(self.runs)

    def __len__(self):
        return len(self.runs)
    
#TODO: Place matched filter search in its own module

#This class stores the output of a matched filter search.
class FilterSearchResult(Experiment):
    def __init__(self, output_directory: str, name: Optional[str] = None):
        super().__init__(name)
        #directory to store search result (pickle files)
        self.output_directory = output_directory
        pass
        



    