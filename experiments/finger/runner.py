import os
import numpy as np
from src.simulator import simulate_finger, make_step_fn, make_step_fn_pusher
from src.visualiser import visualise_finger
from experiments.finger.config import config, mj_model, mj_data, mjx_model, mjx_data

def main():
    print("Running Finger Experiment ...")


    step_fn = make_step_fn_pusher(mjx_model, mjx_data)

    """
    Find out how many 
    """
    states = simulate_finger(mjx_data=mjx_data,
                             num_steps=config.steps,
                             step_function=step_fn)

    # convert the states and jacobians to numpy arrays
    states = np.array(states)

    # Visualise the trajectory
    print("Visualising the trajectory ...")
    visualise_finger(states, mj_data, mj_model)


if __name__ == "__main__":
    main()