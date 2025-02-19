import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os

from src.simulator import simulate, make_step_fn, make_step_fn_fd
from src.visualiser import visualise_traj_generic
from experiments.one_bounce.config import Config

def main():
    print("Running One Bounce Ball Experiment ...")
    config = Config()

    # --- Load model and data ---
    xml_path = "/Users/hashim/Desktop/GradientExperiments/models/ball.xml"
    mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # --- Initialize state ---
    mjx_data = mjx_data.replace(
        qpos=jnp.array([config.init_pos[0], 0.0, config.init_pos[1], 1.0, 0.0, 0.0, 0.0]),
        # p_x, p_y, p_z, quat_w, quat_x, quat_y, quat_z
        qvel=jnp.array([config.init_vel[0], 0.0, config.init_vel[1], 0.0, 0.0, 0.0])  # v_x, v_y, v_z, w_x, w_y, w_z
    )

    step_fn_fd = make_step_fn(mjx_model, mjx_data)
    step_fn = make_step_fn(mjx_model, mjx_data)

    states, jacobians = simulate(mjx_data=mjx_data,
                                 num_steps=config.steps,
                                 step_function=step_fn)

    # Save the states and jacobians into files for later use
    current_directory = os.path.dirname(os.path.realpath(__file__))
    stored_data_directory = os.path.join(current_directory, 'stored_data')
    np.save(os.path.join(stored_data_directory, 'states.npy'), states)
    np.save(os.path.join(stored_data_directory, 'jacobians.npy'), jacobians)

    print("States and Jacobians saved to stored_data directory")

if __name__ == "__main__":
    main()