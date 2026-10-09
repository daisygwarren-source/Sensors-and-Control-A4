"""Build the Track B scene: UR5e + Robotiq 2F-85 + wrist RGB-D camera + table + target object.
 
Uses MjSpec so the Menagerie files stay untouched. Run directly to export scene.xml.
"""
from pathlib import Path
import numpy as np
import mujoco
 
HERE = Path(__file__).resolve().parent
UR5E_XML = HERE / "universal_robots_ur5e" / "scene.xml"
GRIPPER_XML = HERE / "robotiq_2f85" / "2f85.xml"
 
TABLE_POS = np.array([0.5, 0.0, 0.0])   # table centre (x forward of robot base)
TABLE_HALF = np.array([0.3, 0.4, 0.02])  # half-sizes of the tabletop slab
TABLE_TOP_Z = 0.0 + 2 * TABLE_HALF[2]    # tabletop sits on the floor -> top at z=0.04
OBJ_HALF = 0.02                          # 4 cm cube
 
 
def build(object_xy=(0.5, 0.0)) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # gripper model expects these
    spec.option.impratio = 10
    for k in list(spec.keys):
        spec.delete(k)  # old keyframe no longer matches qpos size once we add things
 
    # --- Gripper on the flange -------------------------------------------
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    flange = spec.site("attachment_site")
    spec.attach(gripper, site=flange, prefix="gripper/")
 
    # --- Wrist RGB-D camera (eye-in-hand) ----------------------------------
    # Mounted on wrist_3_link, offset sideways from the gripper, looking along the tool axis.
    wrist = spec.body("wrist_3_link")
    wrist.add_camera(
        name="wrist_cam",
        pos=[0.0, 0.10, 0.08],      # in wrist_3_link frame
        xyaxes=[1, 0, 0, 0, 0, 1],  # camera looks along -z_cam = +y of wrist_3 (tool axis)
        fovy=58,                    # ~RealSense D435 colour vertical FOV
    )
 
    world = spec.worldbody
 
    # --- Fixed overview camera (for video / debugging) ---------------------
    world.add_camera(name="overview", pos=[1.6, -1.2, 1.2], xyaxes=[0.6, 0.8, 0, -0.35, 0.26, 0.9])
 
    # --- Table ---------------------------------------------------------------
    table = world.add_body(name="table", pos=TABLE_POS)
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=TABLE_HALF,
                   pos=[0, 0, TABLE_HALF[2]], rgba=[0.55, 0.4, 0.3, 1])
 
    # --- Target object (free body, position randomised per trial) ----------
    obj = world.add_body(name="target",
                         pos=[object_xy[0], object_xy[1], TABLE_TOP_Z + OBJ_HALF + 0.001])
    obj.add_freejoint(name="target_free")
    obj.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[OBJ_HALF] * 3,
                 rgba=[0.9, 0.1, 0.1, 1], mass=0.05, friction=[1.0, 0.01, 0.001])
 
    # --- Drop-off zone (visual only) --------------------------------------
    world.add_geom(name="drop_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.06, 0.001, 0],
                   pos=[0.4, 0.3, TABLE_TOP_Z + 0.001], rgba=[0.1, 0.8, 0.1, 0.5],
                   contype=0, conaffinity=0)
    return spec
 
 
if __name__ == "__main__":
    spec = build()
    model = spec.compile()
    out = Path(__file__).parent / "scene.xml"
    out.write_text(spec.to_xml())
    print(f"nq={model.nq} nu={model.nu} cams={[model.camera(i).name for i in range(model.ncam)]}")
    print("actuators:", [model.actuator(i).name for i in range(model.nu)])