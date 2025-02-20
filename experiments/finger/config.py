from dataclasses import dataclass
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# --- Configuration ---
@dataclass
class Config:
    simulation_time: float = 1.0    # Duration of simulation
    steps: int = 1000               # Number of simulation steps
    init_pos: jnp.ndarray = jnp.array([-1.2, 0.0, 0.0])
    init_vel: jnp.ndarray = jnp.zeros(3)

# --- Load model and data ---
xml_path = "/Users/hashim/Desktop/GradientExperiments/models/finger.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# --- Initialize state ---
config = Config()

mjx_data.replace(
    qpos=config.init_pos,
    qvel=config.init_vel
)

print(mjx_data.qpos)
print(mjx_data.qvel)

