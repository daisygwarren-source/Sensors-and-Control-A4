import numpy as np
import mujoco


class IKSolver:

    def __init__(self, site_name="gripper/pinch"):
        self.site_name = site_name

    def solve(
        self,
        model,
        data,
        target_pos,
        q_init,
        iterations=50,
        damping=1e-2,
        tol=1e-3,
    ):

        q = q_init.copy()

        site_id = model.site(self.site_name).id

        for _ in range(iterations):

            data.qpos[:6] = q
            mujoco.mj_forward(model, data)

            current = data.site(self.site_name).xpos.copy()

            error = target_pos - current

            if np.linalg.norm(error) < tol:
                break

            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))

            mujoco.mj_jacSite(
                model,
                data,
                Jp,
                Jr,
                site_id,
            )

            J = Jp[:, :6]

            dq = (
                J.T
                @ np.linalg.solve(
                    J @ J.T + damping * np.eye(3),
                    error,
                )
            )

            dq = np.clip(dq, -0.05, 0.05)

            q += dq

        return q