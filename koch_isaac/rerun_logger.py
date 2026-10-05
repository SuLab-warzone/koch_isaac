"""Optional live grasp diagnostics; importing this module does not start Rerun."""
import warnings

import numpy as np


class RerunLogger:
    """Stream one environment to a viewer without creating an RRD or rollout file.

    Logging runs at the control rate, before automatic reset. Normal forces are
    measured on the finger bodies against the box, not the total upward force on
    the box. Only the selected environment's tensors are copied to the CPU.
    """

    def __init__(self, env, env_id=0, *, recording=None):
        # Keep Rerun optional: normal evaluation/training does not import its SDK.
        import rerun as rr
        import rerun.blueprint as rrb

        if not 0 <= env_id < env.num_envs:
            raise ValueError("Rerun environment index is outside the scene")
        self.rr = rr
        self.env_id = env_id
        self.joint_id = env.scene["robot"].joint_names.index("follower_joint_gripper")
        self.recording = recording if recording is not None else rr.RecordingStream("koch_grasp_debug")
        # Supplying a recording allows validation with an in-memory sink, without a GUI.
        if recording is None:
            self.recording.spawn(memory_limit="1GiB", hide_welcome_screen=True)
        self.recording.send_blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name=f"Front camera • env {env_id}", origin="/camera"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Finger normal force / ideal grip [N]", origin="/forces"),
                    rrb.TimeSeriesView(name="Gripper angle [rad]", origin="/gripper"),
                    rrb.TimeSeriesView(name="Box bottom above table [m]", origin="/box"),
                ),
            )
        )
        # Distinct colors make a missing moving-jaw contact easy to recognize.
        for path, label, color in (
            ("forces/static", "Static finger", [50, 140, 255]),
            ("forces/moving", "Moving finger", [255, 140, 40]),
            ("forces/ideal_static", "Ideal per finger (static friction)", [80, 200, 100]),
            ("forces/ideal_sliding", "Ideal per finger (sliding friction)", [200, 170, 80]),
            ("forces/contact_threshold", "Contact detection threshold", [160, 160, 160]),
        ):
            self.recording.log(path, rr.SeriesLines(names=label, colors=color), static=True)
        self.recording.log("notes", rr.TextDocument(
            "Forces: box-filtered normal resultants on each finger; no tangential friction. "
            "Reference N=mg/(2*mu): ideal equal opposing side pinch, no acceleration; "
            "uses configured box friction, not combined finger/box friction. "
            "Sampled at control rate; transient physics-tick contacts may be missed."
        ), static=True)

    def log(self, env, lift):
        """Called after force sampling, before any completed environment is reset."""
        rr, rec, i, j = self.rr, self.recording, self.env_id, self.joint_id
        # A global simulation timeline stays monotonic across episode resets.
        rec.set_time("sim_time", duration=env.common_step_counter * env.step_dt)
        rec.set_time("step", sequence=int(env.common_step_counter))
        force = env.contact_metrics.last_force[i].detach().cpu().numpy()  # [2, xyz], N
        magnitude = np.linalg.norm(force, axis=-1)
        for name, value in zip(("static", "moving"), magnitude):
            rec.log(f"forces/{name}", rr.Scalars(float(value)))
        rec.log("forces/contact_threshold", rr.Scalars(env.grasp_force))

        # Plot per-finger grip references on the normal-force chart. Weight itself
        # is an upward support requirement and is deliberately not a normal-force curve.
        metrics = env.contact_metrics
        weight = metrics.mass_kg * metrics.gravity_m_s2
        for name, mu in (("ideal_static", metrics.static_friction),
                         ("ideal_sliding", metrics.dynamic_friction)):
            if mu > 0:
                rec.log(f"forces/{name}", rr.Scalars(weight / (2 * mu)))

        robot = env.scene["robot"]
        rec.log("gripper/target", rr.Scalars(robot.data.joint_pos_target.torch[i, j].item()))
        rec.log("gripper/measured", rr.Scalars(robot.data.joint_pos.torch[i, j].item()))
        rec.log("box/bottom_height", rr.Scalars(lift[i].item()))
        rec.log("box/grasp_height_threshold", rr.Scalars(env.grasp_height))
        # Read the rendered frame at the same pre-reset point as joint/contact data.
        # Policy modes enable this camera before constructing the environment.
        rgb = env.scene["front_camera"].data.output["rgb"][i, ..., :3]
        rec.log("camera/front", rr.Image(rgb.detach().cpu().numpy()))

    def close(self):
        # Bound shutdown time if the user has closed the viewer during evaluation.
        try:
            self.recording.flush(timeout_sec=2.0)
        except Exception as error:
            warnings.warn(f"Rerun flush failed: {error}", RuntimeWarning)
        finally:
            self.recording.disconnect()
