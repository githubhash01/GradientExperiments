import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd
from src.config import Config
from src.visualiser import visualise_traj_generic, rf

config = Config()

# --- Load the model ---
xml_path = "/Users/hashim/Desktop/GradientExperiments/src/ball.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Set initial position and velocity
mjx_data = mjx_data.replace(
    qpos=jnp.array([config.init_pos[0], 0.0, config.init_pos[1], 1.0, 0.0, 0.0, 0.0]),  # [X, Y=0, Z, ...]
    qvel=jnp.array([config.init_vel[0], 0.0, config.init_vel[1], 0.0, 0.0, 0.0])  # [vx, vy=0, vz, ...]
)


# Define simulation step function
@jax.jit
def step_fn(dx):
    dx = mjx.step(mjx_model, dx)
    state = jnp.concatenate([dx.qpos, dx.qvel])
    return dx, state


@jax.jit
def step_state(state):
    # mjx_model.nq gives the length of each component of the state vector
    nq = mjx_model.nq

    # Split state into q_pos and q_vel
    q_pos = state[:nq]
    q_vel = state[nq:]

    dx = mjx_data.replace(qpos=q_pos, qvel=q_vel)

    # Now step the simulation using the existing step_fn.
    dx_next = mjx.step(mjx_model, dx)
    next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])

    return dx_next, next_state

def f(state):
    # We only care about the next state, not dx_next.
    _, next_state = step_state(state)
    return next_state

def simulate(num_steps):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state] # store the states
    jacobians_s = [] # store the jacobians of the states

    for _ in range(num_steps):

        # Find the jacobian of the state
        jac = jax.jacobian(f)(state)
        jacobians_s.append(jac)

        dx, state = step_state(state)
        states.append(np.array(state))

    return states



def main():
    print("------------ Position-based Dynamics (MJX)-----------")



    states = simulate(num_steps=config.steps)

    # convert states into an array
    states = np.array(states)
    # Visualize the trajectory
    visualise_traj_generic(states, mj_data, mj_model)

    # Extract final position
    final_state = states[-1]
    final_pos = final_state[:3]
    print("Final position of the ball: ", [rf(x) for x in final_pos])

if __name__ == "__main__":
    main()