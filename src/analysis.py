"""
Main file for doing analysis on the data generated from the experiments
"""

import os
import numpy as np
from src.visualiser import visualise_traj_generic
from src.utils import print_state_jacobian

def main():

    print("Running analysis on the data ...")

    # Load the data
    stored_data_directory = "/Users/hashim/Desktop/GradientExperiments/experiments/one_bounce/stored_data"
    states = np.load(os.path.join(stored_data_directory, 'states.npy'))
    jacobians = np.load(os.path.join(stored_data_directory, 'jacobians.npy'))

    # Perform analysis on the data
    print_state_jacobian(jacobians[0])

    print("Analysis completed")