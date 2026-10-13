def controller(model, data):

    q = data.qpos[:6].copy()

    cube_world = data.body("target").xpos.copy()

    # target 15 cm above cube
    target = cube_world + np.array([0.0, 0.0, 0.15])

    q_des = ik.solve(
        model,
        data,
        target,
        q,
        iterations=10,
    )

    # smooth movement towards IK solution
    alpha = 0.05
    data.ctrl[:6] = (1 - alpha) * q + alpha * q_des

    # keep gripper open
    data.ctrl[6] = 0