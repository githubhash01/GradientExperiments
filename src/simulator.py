import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd  # Forward-mode autodiff

# --- Auto-Differentiation ---
def make_step_fn(model, mjx_data):

    @jax.jit
    def step_fn(state):
        nq = model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        dx_next = mjx.step(model, dx)
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    return step_fn

# --- Finite Differences ---
def make_step_fn_fd(mjx_model, mjx_data):
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

# --- Compute Jacobians Functions using JAX ---
#step_fn = make_step_fn(mjx_model)
#jac_fn = jax.jit(jacfwd(step_fn))  # Forward-mode is safer for loops

#step_fn_fd = make_step_fn_fd()
#jac_fn_rev = jax.jit(jax.jacrev(step_fn_fd))

def simulate(mjx_data, num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    # define the backwards jacobian functio
    jac_fn = jax.jit(jacfwd(step_function))  # Forward-mode is safer for loopsn
    #jac_fn_rev = jax.jit(jax.jacrev(step_function))

    for _ in range(num_steps):
        # Compute Jacobian BEFORE stepping (gradient of NEXT state w.r.t. CURRENT state)
        J_s = jac_fn(state)
        #J_s = jac_fn_rev(state)

        state_jacobians.append(J_s)

        # Step forward
        state = step_function(state)
        states.append(np.array(state))

    return states, state_jacobians


