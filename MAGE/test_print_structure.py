#!/usr/bin/env python3
"""
Simple test script to demonstrate the print_structure method.
This creates a mock experiment structure and shows the ASCII tree output.
"""

import os
import sys

# Add the utilities directory to the path so we can import dataStream
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))

from dataStream import Experiment, Run, DataFile

def test_print_structure():
    """Test the print_structure method with a mock experiment."""

    # Create a mock experiment
    exp = Experiment("test_experiment", "/fake/path", None, read_metadata_on_init=False)

    # Create some mock runs
    run1 = Run(exp, "run1")
    run2 = Run(exp, "run2")

    # Add runs to experiment
    exp.add_run(run1)
    exp.add_run(run2)

    # Create mock data files
    file1 = DataFile(run1, "/fake/path/run1/run1-1.hdf5")
    file2 = DataFile(run1, "/fake/path/run1/run1-2.hdf5")
    file3 = DataFile(run2, "/fake/path/run2/run2-1.hdf5")

    # Add files to runs
    run1.add_file(file1)
    run1.add_file(file2)
    run2.add_file(file3)

    # Print the structure
    print("Testing print_structure method:")
    print("=" * 40)
    exp.print_structure()

if __name__ == "__main__":
    test_print_structure()