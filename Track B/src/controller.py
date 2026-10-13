def controller(model, data):

    q = data.qpos[:6].copy()

    cube_world = data.body("target").xpos.copy()

    print("target:", cube_world)

    target = np.array([
        cube_world[0],
        cube_world[1],
        cube_world[2] + 0.15,
    ])

    q_des = ik.solve(
        model,
        data,
        target,
        q,
    )

    data.ctrl[:6] = q_des