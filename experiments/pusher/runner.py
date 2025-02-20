import os
import numpy as np
from src.simulator import simulate_pusher, make_step_fn_pusher
from src.visualiser import visualise_pusher
from experiments.pusher.config import config, mj_model, mj_data, mjx_model, mjx_data

def main():
    print("Running Pusher Experiment ...")

    step_fn = make_step_fn_pusher(mjx_model, mjx_data)

    states = simulate_pusher(mjx_data=mjx_data,
                                 num_steps=config.steps,
                                 step_function=step_fn)

    # convert the states and jacobians to numpy arrays
    states = np.array(states)

    # Visualise the trajectory
    print("Visualising the trajectory ...")

    visualise_pusher(states, mj_data, mj_model)

if __name__ == "__main__":
    main()
