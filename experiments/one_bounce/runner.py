import os
import numpy as np
from src.simulator import simulate, make_step_fn, make_step_fn_fd
from src.visualiser import visualise_traj_generic
from experiments.one_bounce.config import config, mj_model, mj_data, mjx_model, mjx_data

def main():
    print("Running One Bounce Ball Experiment ...")


    step_fn_fd = make_step_fn(mjx_model, mjx_data)
    step_fn = make_step_fn(mjx_model, mjx_data)

    states, jacobians = simulate(mjx_data=mjx_data,
                                 num_steps=config.steps,
                                 step_function=step_fn_fd)

    # convert the states and jacobians to numpy arrays
    states = np.array(states)
    jacobians = np.array(jacobians)

    # Visualise the trajectory
    print("Visualising the trajectory ...")
    visualise_traj_generic(states, mj_data, mj_model)

    # Save the states and jacobians into files for later use
    current_directory = os.path.dirname(os.path.realpath(__file__))
    stored_data_directory = os.path.join(current_directory, 'stored_data')
    np.save(os.path.join(stored_data_directory, 'states_fd.npy'), states)
    np.save(os.path.join(stored_data_directory, 'jacobians_fd.npy'), jacobians)

    print("States and Jacobians saved to stored_data directory")

if __name__ == "__main__":
    main()