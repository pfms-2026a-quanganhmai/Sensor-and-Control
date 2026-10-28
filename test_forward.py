import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    r"robotis_mujoco_menagerie\robotis_tb3\scene_turtlebot3_waffle_pi.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 8.0:
        step_start = time.time()

        data.ctrl[0] = 4.0
        data.ctrl[1] = 4.0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Keep simulation close to real time
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    data.ctrl[:] = 0