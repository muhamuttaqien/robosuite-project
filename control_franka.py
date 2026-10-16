import cv2
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


# ============================================================
# Configuration
# ============================================================

WINDOW_NAME = "Franka Panda - Keyboard Control"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

MOVE_SPEED = 0.4
ROTATION_SPEED = 0.3
GRIPPER_ROTATION_SPEED = 0.3

CONTROL_FREQUENCY = 20


# ============================================================
# Three-cube environment
# ============================================================

class ThreeCubeLift(Lift):
    """Franka Panda with red, green, and blue cubes."""

    def _load_model(self):
        super()._load_model()

        arena = self.model.mujoco_arena

        self.green_cube = BoxObject(
            name="green_cube",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0.1, 0.9, 0.2, 1.0],
        )

        self.blue_cube = BoxObject(
            name="blue_cube",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0.1, 0.3, 1.0, 1.0],
        )

        self.placement_initializer = UniformRandomSampler(
            name="ThreeCubeSampler",
            mujoco_objects=[
                self.cube,
                self.green_cube,
                self.blue_cube,
            ],
            x_range=[-0.15, 0.15],
            y_range=[-0.15, 0.15],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[
                robot.robot_model for robot in self.robots
            ],
            mujoco_objects=[
                self.cube,
                self.green_cube,
                self.blue_cube,
            ],
        )


# ============================================================
# Local gripper-axis rotation
# ============================================================

def get_gripper_axis_world(env):
    """
    Return the gripper's local Z-axis in world coordinates.

    Uses MuJoCo's current end-effector orientation matrix
    whenever available.
    """

    robot = env.robots[0]

    # Robosuite stores hand orientation separately for each arm.
    hand_orn = robot._hand_orn

    if isinstance(hand_orn, dict):
        arm_name = (
            "right"
            if "right" in hand_orn
            else next(iter(hand_orn))
        )
        hand_orn = hand_orn[arm_name]

    orientation = np.asarray(
        hand_orn,
        dtype=np.float64,
    )

    if orientation.shape == (3, 3):
        rotation_matrix = orientation

    elif orientation.size == 9:
        rotation_matrix = orientation.reshape(3, 3)

    elif orientation.shape == (4,):
        # Quaternion fallback.
        # This assumes MuJoCo's wxyz convention.
        from scipy.spatial.transform import Rotation

        quat_wxyz = orientation

        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ])

        rotation_matrix = Rotation.from_quat(
            quat_xyzw
        ).as_matrix()

    else:
        raise ValueError(
            "Unsupported end-effector orientation "
            f"shape: {orientation.shape}"
        )

    local_z_axis = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    world_axis = rotation_matrix @ local_z_axis

    axis_norm = np.linalg.norm(world_axis)

    if axis_norm < 1e-8:
        raise RuntimeError(
            "Invalid gripper rotation axis."
        )

    return world_axis / axis_norm


# ============================================================
# Print keyboard instructions
# ============================================================

def print_controls():
    print()
    print("=" * 58)
    print("       FRANKA PANDA | KEYBOARD TELEOPERATION")
    print("=" * 58)
    print()
    print("  ARM MOVEMENT                  END-EFFECTOR ROTATION")
    print("  -------------------------     -------------------------")
    print("  W / S   Forward / Backward    I / K   Roll + / -")
    print("  A / D   Left / Right          J / L   Pitch + / -")
    print("  R / F   Up / Down             U / O   Yaw + / -")
    print()
    print("  GRIPPER CONTROL               SIMULATION")
    print("  -------------------------     -------------------------")
    print("  Z / X   Twist + / -           Q       Reset robot & cubes")
    print("  SPACE   Open / Close          Ctrl+C  Quit (terminal)")
    print()
    print("-" * 58)
    print("  1. Click the wrist-camera window to activate controls.")
    print("  2. Press a movement key to move the robot.")
    print("  3. Use Z / X to twist the gripper around its own axis.")
    print("  4. Use SPACE to grasp or release an object.")
    print("=" * 58)
    print()


# ============================================================
# Main simulation
# ============================================================

def main():
    env = None
    gripper_closed = False

    try:
        env = ThreeCubeLift(
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["robot0_eye_in_hand"],
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            camera_depths=False,
            reward_shaping=True,
            ignore_done=True,
            control_freq=CONTROL_FREQUENCY,
        )

        obs = env.reset()

        # Interactive MuJoCo camera.
        env.viewer.set_camera(camera_id=-1)

        if env.action_dim != 7:
            raise RuntimeError(
                "Expected 7D OSC_POSE action space, "
                f"but got {env.action_dim}."
            )

        # Small wrist-camera window.
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )

        print_controls()

        # ====================================================
        # Simulation loop
        # ====================================================

        while True:

            # ------------------------------------------------
            # Wrist-camera display
            # ------------------------------------------------

            wrist_image = obs[
                "robot0_eye_in_hand_image"
            ]

            wrist_image = cv2.flip(
                wrist_image,
                0,
            )

            wrist_image = cv2.cvtColor(
                wrist_image,
                cv2.COLOR_RGB2BGR,
            )

            cv2.imshow(
                WINDOW_NAME,
                wrist_image,
            )

            # ------------------------------------------------
            # Keyboard input
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            action = np.zeros(
                7,
                dtype=np.float64,
            )

            # ------------------------------------------------
            # Cartesian translation
            # ------------------------------------------------

            if key == ord("w"):
                action[0] = MOVE_SPEED

            elif key == ord("s"):
                action[0] = -MOVE_SPEED

            elif key == ord("a"):
                action[1] = MOVE_SPEED

            elif key == ord("d"):
                action[1] = -MOVE_SPEED

            elif key == ord("r"):
                action[2] = MOVE_SPEED

            elif key == ord("f"):
                action[2] = -MOVE_SPEED

            # ------------------------------------------------
            # End-effector rotation
            # ------------------------------------------------

            elif key == ord("i"):
                action[3] = ROTATION_SPEED

            elif key == ord("k"):
                action[3] = -ROTATION_SPEED

            elif key == ord("j"):
                action[4] = ROTATION_SPEED

            elif key == ord("l"):
                action[4] = -ROTATION_SPEED

            elif key == ord("u"):
                action[5] = ROTATION_SPEED

            elif key == ord("o"):
                action[5] = -ROTATION_SPEED

            # ------------------------------------------------
            # Gripper local-axis rotation
            # ------------------------------------------------

            elif key in (ord("z"), ord("x")):

                direction = (
                    1.0
                    if key == ord("z")
                    else -1.0
                )

                axis_world = get_gripper_axis_world(env)

                action[3:6] = (
                    direction
                    * GRIPPER_ROTATION_SPEED
                    * axis_world
                )

            # ------------------------------------------------
            # Gripper open / close
            # ------------------------------------------------

            elif key == ord(" "):

                gripper_closed = not gripper_closed

                print(
                    "[GRIPPER]",
                    "CLOSED" if gripper_closed else "OPEN",
                )

            # ------------------------------------------------
            # Reset simulation
            # ------------------------------------------------

            elif key == ord("q"):

                print("[SIMULATION] Resetting robot and cubes...")

                obs = env.reset()

                gripper_closed = False

                env.render()

                continue

            # ------------------------------------------------
            # Apply gripper command
            # ------------------------------------------------

            action[6] = (
                1.0
                if gripper_closed
                else -1.0
            )

            # ------------------------------------------------
            # Step simulation
            # ------------------------------------------------

            obs, reward, done, info = env.step(action)

            env.render()

    except KeyboardInterrupt:
        print("\n[SIMULATION] Stopped by user.")

    finally:
        if env is not None:
            env.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
