import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd  # Forward-mode autodiff
from src.config import Config
from src.visualiser import visualise_traj_generic, rf

config = Config()

# --- Load model and data ---
xml_path = "/Users/hashim/Desktop/GradientExperiments/src/ball.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# --- Initialize state ---
mjx_data = mjx_data.replace(
    qpos=jnp.array([config.init_pos[0], 0.0, config.init_pos[1], 1.0, 0.0, 0.0, 0.0]),
    qvel=jnp.array([config.init_vel[0], 0.0, config.init_vel[1], 0.0, 0.0, 0.0])
)


# --- Define differentiable step function ---
@jax.jit
def step_fn(state):
    nq = mjx_model.nq
    qpos, qvel = state[:nq], state[nq:]
    dx = mjx_data.replace(qpos=qpos, qvel=qvel)
    dx_next = mjx.step(mjx_model, dx)
    next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
    return next_state


# --- Compute Jacobians using forward-mode autodiff ---
jac_fn = jax.jit(jacfwd(step_fn))  # Forward-mode is safer for loops


def simulate(num_steps):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    for _ in range(num_steps):
        # Compute Jacobian BEFORE stepping (gradient of NEXT state w.r.t. CURRENT state)
        J_s = jac_fn(state)
        state_jacobians.append(J_s)

        # Step forward
        state = step_fn(state)
        states.append(np.array(state))

    return states, state_jacobians


def main():
    print("------------ Position-based Dynamics (MJX)-----------")
    states, jacobians = simulate(num_steps=config.steps)
    visualise_traj_generic(np.array(states), mj_data, mj_model)


if __name__ == "__main__":
    main()