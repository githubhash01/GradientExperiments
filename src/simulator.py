import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd  # Forward-mode autodiff
from src.config import Config
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

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
    qpos=jnp.array([config.init_pos[0], 0.0, config.init_pos[1], 1.0, 0.0, 0.0, 0.0]), # p_x, p_y, p_z, quat_w, quat_x, quat_y, quat_z
    qvel=jnp.array([config.init_vel[0], 0.0, config.init_vel[1], 0.0, 0.0, 0.0]) # v_x, v_y, v_z, w_x, w_y, w_z
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


def make_step_fn_fd():
    epsilon = 1e-5

    # This function performs the actual step computation.
    @jax.jit
    def _step_fn(state):
        nq = mjx_model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        dx_next = mjx.step(mjx_model, dx)
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    # Define the custom_vjp-decorated function.
    @jax.custom_vjp
    def step_fn(state):
        return _step_fn(state)

    # Forward pass: compute the output and return the input (or any auxiliary data)
    def step_fn_fwd(s):
        f_s = _step_fn(s)
        return f_s, s  # saving s for use in backward pass

    # Backward pass: given the saved s and the incoming cotangent,
    # compute the vector-Jacobian product using finite differences.
    def step_fn_bwd(s, cotangent):
        f_s = _step_fn(s)  # compute baseline output once
        grad = []
        for j in range(s.shape[0]):
            # Create a perturbation along the j-th coordinate.
            e_j = jnp.zeros_like(s).at[j].set(1.0)
            # Evaluate the function at the perturbed state.
            f_perturbed = _step_fn(s + epsilon * e_j)
            # Approximate the partial derivative for coordinate j.
            diff = (f_perturbed - f_s) / epsilon
            # Multiply by the cotangent to get the j-th component of the VJP.
            grad_j = jnp.vdot(cotangent, diff)
            grad.append(grad_j)
        grad = jnp.stack(grad)
        return (grad,)

    # Register the forward and backward functions with the custom_vjp mechanism.
    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn

# Defining the jacobian function using reverse-mode custom VJP with finite differences
step_fn_fd = make_step_fn_fd()
jac_fn_rev = jax.jit(jax.jacrev(step_fn_fd))



def simulate(num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    for _ in range(num_steps):
        # Compute Jacobian BEFORE stepping (gradient of NEXT state w.r.t. CURRENT state)
        #J_s = jac_fn(state)
        J_s = jac_fn_rev(state)

        state_jacobians.append(J_s)

        # Step forward
        state = step_function(state)
        states.append(np.array(state))

    return states, state_jacobians


def main():
    print("------------ Position-based Dynamics (MJX)-----------")
    states, jacobians = simulate(num_steps=config.steps, step_function=step_fn_fd)
    visualise_traj_generic(np.array(states), mj_data, mj_model)

    # Suppose state_jacobians is a list of Jacobian matrices (each is a 2D array)
    jacobian_array = np.array(jacobians)  # Shape: (num_steps, 13, 13)

    # Save to a file (binary format)
    np.save('jacobians.npy', jacobian_array)


if __name__ == "__main__":
    main()