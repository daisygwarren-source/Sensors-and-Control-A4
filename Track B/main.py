import mujoco # we will be using mujuco for the simulator


#set up meshes here
model = mujoco.MjModel.from_xml_path("myrobot.urdf")

data = mujoco.MjData(model)

# Stepping physics directly in the control loop
# data.ctrl[0] = torque_cmd
mujoco.mj_step(model, data)
mujoco.mj_saveLastXML("your_robot.xml", model)