from dataclasses import dataclass
import jax.numpy as jnp

# --- Configuration ---
@dataclass
class Config:
    simulation_time: float = 1.0    # Duration of simulation
    steps: int = 10               # Number of simulation steps
    elasticity: float = 1.0         # Coefficient of restitution
    radius: float = 0.1             # Radius of the ball
    init_pos: jnp.ndarray = jnp.array([-1.0, 1.0])  # Initial position of the ball
    init_vel: jnp.ndarray = jnp.array([2.0, -2.0])  # Initial velocity of the ball
    ctrl_input: jnp.ndarray = jnp.array([0.0, 0.0])
    customized_kn: float = 1.0 # for soft model of the ball