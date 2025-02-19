from mujoco import viewer
import time
import mujoco
import numpy as np

def visualise_traj_generic(x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):

    with viewer.launch_passive(m, d) as v:
        x = np.array(x)
        for i in range(x.shape[0]):
            step_start = time.time()
            qpos = x[i, :m.nq]
            qvel = x[i, m.nq:m.nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

