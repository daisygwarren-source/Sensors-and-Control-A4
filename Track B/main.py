import mujoco # we will be using mujuco for the simulator

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Stepping physics directly in the control loop
data.ctrl[0] = torque_cmd
mujoco.mj_step(model, data)