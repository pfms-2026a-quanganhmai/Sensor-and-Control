import math
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation.ekf import EKF

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "robotis_tb3"
    / "scene_turtlebot3_waffle_pi.xml"
)

SIMULATION_TIME = 45.0

# TurtleBot3 geometry
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.287

# Navigation target
TARGET_X = 1.0
TARGET_Y = 0.5

POSITION_TOLERANCE = 0.05

# Temporary controller gains
K_DISTANCE = 0.8
K_HEADING = 1.8

MAX_LINEAR_VELOCITY = 0.12
MAX_ANGULAR_VELOCITY = 0.8

CAMERA_UPDATE_INTERVAL = 0.10  # 10 Hz

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOVY = 60.0

# Plane is 0.24 m wide, but the actual black marker
# occupies 600 / 800 = 75% of the texture.
ARUCO_MARKER_SIZE = 0.18  # m

# Known ArUco marker pose in the MuJoCo world
MARKER_WORLD_POSITION = np.array([
    1.50,
    0.00,
    0.20,
])

# Marker orientation in world coordinates.
# Matches:
# euler="0 -1.5708 0"
MARKER_WORLD_ROTATION = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
])

# Camera position relative to TurtleBot base
CAMERA_POSITION_IN_ROBOT = np.array([
    0.10,
    0.00,
    0.12,
])

def wrap_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def body_velocity_to_wheel_speed(
    linear_velocity,
    angular_velocity,
):
    left = (
        linear_velocity
        - angular_velocity
        * WHEEL_SEPARATION / 2.0
    ) / WHEEL_RADIUS

    right = (
        linear_velocity
        + angular_velocity
        * WHEEL_SEPARATION / 2.0
    ) / WHEEL_RADIUS

    return left, right


def get_robot_pose(data):
    # MuJoCo ground truth.
    # Only used for evaluation and
    # temporary simulated camera measurement.

    x = data.qpos[0]
    y = data.qpos[1]

    qw = data.qpos[3]
    qx = data.qpos[4]
    qy = data.qpos[5]
    qz = data.qpos[6]

    heading = math.atan2(
        2.0 * (
            qw * qz
            + qx * qy
        ),
        1.0 - 2.0 * (
            qy * qy
            + qz * qz
        ),
    )

    return x, y, heading

def aruco_to_robot_pose(rvec, tvec):
    # solvePnP gives:
    # marker frame -> camera frame
    rotation_marker_to_camera, _ = cv2.Rodrigues(
        rvec
    )

    # Camera orientation in world coordinates
    rotation_camera_to_world = (
        MARKER_WORLD_ROTATION
        @ rotation_marker_to_camera.T
    )

    # Camera forward axis is its local +Z axis
    camera_forward_world = (
        rotation_camera_to_world[:, 2]
    )

    robot_heading = math.atan2(
        camera_forward_world[1],
        camera_forward_world[0],
    )

    robot_heading = wrap_angle(
        robot_heading
    )

    # Marker position relative to camera
    marker_in_camera = tvec.reshape(3)

    # Camera world position
    camera_world_position = (
        MARKER_WORLD_POSITION
        - rotation_camera_to_world
        @ marker_in_camera
    )

    # Robot heading rotation
    cos_h = math.cos(robot_heading)
    sin_h = math.sin(robot_heading)

    rotation_robot_to_world = np.array([
        [cos_h, -sin_h, 0.0],
        [sin_h,  cos_h, 0.0],
        [0.0,    0.0,   1.0],
    ])

    camera_offset_world = (
        rotation_robot_to_world
        @ CAMERA_POSITION_IN_ROBOT
    )

    robot_world_position = (
        camera_world_position
        - camera_offset_world
    )

    return (
        robot_world_position[0],
        robot_world_position[1],
        robot_heading,
    )

def waypoint_controller(
    x,
    y,
    heading,
):
    dx = TARGET_X - x
    dy = TARGET_Y - y

    distance = math.hypot(
        dx,
        dy,
    )

    if distance < POSITION_TOLERANCE:
        return 0.0, 0.0, True

    desired_heading = math.atan2(
        dy,
        dx,
    )

    heading_error = wrap_angle(
        desired_heading - heading
    )

    linear_velocity = min(
        K_DISTANCE * distance,
        MAX_LINEAR_VELOCITY,
    )

    # Slow forward motion if heading
    # error becomes large.
    linear_velocity *= max(
        0.0,
        math.cos(heading_error),
    )

    angular_velocity = (
        K_HEADING
        * heading_error
    )

    angular_velocity = max(
        -MAX_ANGULAR_VELOCITY,
        min(
            angular_velocity,
            MAX_ANGULAR_VELOCITY,
        ),
    )

    return (
        linear_velocity,
        angular_velocity,
        False,
    )


def main():
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )

    aruco_dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    aruco_parameters = (
        cv2.aruco.DetectorParameters()
    )

    aruco_detector = cv2.aruco.ArucoDetector(
        aruco_dictionary,
        aruco_parameters,
    )

    focal_length = (
        CAMERA_HEIGHT
        / (
            2.0
            * math.tan(
                math.radians(CAMERA_FOVY) / 2.0
            )
        )
    )

    camera_matrix = np.array([
        [
            focal_length,
            0.0,
            (CAMERA_WIDTH - 1) / 2.0,
        ],
        [
            0.0,
            focal_length,
            (CAMERA_HEIGHT - 1) / 2.0,
        ],
        [
            0.0,
            0.0,
            1.0,
        ],
    ])

    dist_coeffs = np.zeros((5, 1))

    half_marker = ARUCO_MARKER_SIZE / 2.0

    marker_object_points = np.array([
        [-half_marker,  half_marker, 0.0],
        [ half_marker,  half_marker, 0.0],
        [ half_marker, -half_marker, 0.0],
        [-half_marker, -half_marker, 0.0],
    ], dtype=np.float32)

    ekf = EKF()

    # -----------------------------
    # Wheel encoder joint addresses
    # -----------------------------

    left_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "wheel_left",
    )

    right_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "wheel_right",
    )

    left_qpos_address = (
        model.jnt_qposadr[
            left_joint_id
        ]
    )

    right_qpos_address = (
        model.jnt_qposadr[
            right_joint_id
        ]
    )

    previous_left_angle = (
        data.qpos[
            left_qpos_address
        ]
    )

    previous_right_angle = (
        data.qpos[
            right_qpos_address
        ]
    )

    last_camera_update = -CAMERA_UPDATE_INTERVAL
    camera_image = None

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:


        while viewer.is_running():
            step_start = time.time()

            elapsed = data.time

            if elapsed >= SIMULATION_TIME:
                break

            # =================================
            # 1. Wheel encoder measurement
            # =================================

            left_angle = (
                data.qpos[
                    left_qpos_address
                ]
            )

            right_angle = (
                data.qpos[
                    right_qpos_address
                ]
            )

            delta_left_angle = (
                left_angle
                - previous_left_angle
            )

            delta_right_angle = (
                right_angle
                - previous_right_angle
            )

            previous_left_angle = left_angle
            previous_right_angle = right_angle

            delta_left = (
                WHEEL_RADIUS
                * delta_left_angle
            )

            delta_right = (
                WHEEL_RADIUS
                * delta_right_angle
            )

            delta_distance = (
                delta_left
                + delta_right
            ) / 2.0

            delta_heading = (
                delta_right
                - delta_left
            ) / WHEEL_SEPARATION

            # =================================
            # 2. EKF prediction
            # =================================

            ekf.predict(
                delta_distance,
                delta_heading,
            )

            last_pose_update = elapsed

            # =================================
            # 4. Controller uses EKF state
            # =================================

            estimated_x = ekf.x[0]
            estimated_y = ekf.x[1]
            estimated_heading = ekf.x[2]

            (
                linear_velocity,
                angular_velocity,
                reached,
            ) = waypoint_controller(
                estimated_x,
                estimated_y,
                estimated_heading,
            )

            # =================================
            # 5. Convert body command
            #    to wheel actuator commands
            # =================================

            (
                left_speed,
                right_speed,
            ) = body_velocity_to_wheel_speed(
                linear_velocity,
                angular_velocity,
            )

            data.ctrl[0] = left_speed
            data.ctrl[1] = right_speed

            # =================================
            # 6. MuJoCo step
            # =================================

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            # =================================
            # 7. Camera + ArUco detection
            # =================================

            if (
                data.time - last_camera_update
                >= CAMERA_UPDATE_INTERVAL
            ):
                renderer.update_scene(
                    data,
                    camera="front_camera",
                )

                camera_image = renderer.render()

                # ---------------------------------
                # ArUco detection
                # Use original camera orientation
                # ---------------------------------

                gray = cv2.cvtColor(
                    camera_image,
                    cv2.COLOR_RGB2GRAY,
                )

                corners, ids, rejected = (
                    aruco_detector.detectMarkers(gray)
                )

                camera_bgr = cv2.cvtColor(
                    camera_image,
                    cv2.COLOR_RGB2BGR,
                )

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(
                        camera_bgr,
                        corners,
                        ids,
                    )

                    detected_ids = ids.flatten()

                    for marker_corners, marker_id in zip(
                        corners,
                        detected_ids,
                    ):
                        if marker_id == 0:
                            image_points = (
                                marker_corners
                                .reshape(4, 2)
                                .astype(np.float32)
                            )

                            success, rvec, tvec = cv2.solvePnP(
                                marker_object_points,
                                image_points,
                                camera_matrix,
                                dist_coeffs,
                                # 当前 OpenCV 的 IPPE 在近正视平面时产生错误解；
                                # 改用迭代 PnP，并在联调测试中验证图像位姿误差。
                                flags=cv2.SOLVEPNP_ITERATIVE,
                            )

                            if success:
                                distance = float(
                                    np.linalg.norm(tvec)
                                )

                                (
                                    measured_x,
                                    measured_y,
                                    measured_heading,
                                ) = aruco_to_robot_pose(
                                    rvec,
                                    tvec,
                                )

                                # Real camera measurement
                                # enters the EKF here.
                                ekf.update_pose(
                                    measured_x,
                                    measured_y,
                                    measured_heading,
                                )

                                print(
                                    f"ArUco pose | "
                                    f"x={measured_x:.3f}, "
                                    f"y={measured_y:.3f}, "
                                    f"theta="
                                    f"{math.degrees(measured_heading):.1f} deg "
                                    f"| range={distance:.3f} m",
                                    end="\r",
                                )

                # Rotate only the display image
                display_image = cv2.rotate(
                    camera_bgr,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                )

                cv2.imshow(
                    "TurtleBot Front Camera",
                    display_image,
                )

                cv2.waitKey(1)

                last_camera_update = data.time                

            #print(
            #    f"EKF: "
            #    f"x={estimated_x:.2f}, "
            #    f"y={estimated_y:.2f}, "
            #    f"theta={math.degrees(estimated_heading):.1f} deg "
            #    f"| Camera={camera_image.shape if camera_image is not None else 'waiting'}",
            #    end="\r",
            #)

            if reached:
                print(
                    "\nWaypoint reached."
                )
                break

            remaining = (
                model.opt.timestep
                - (
                    time.time()
                    - step_start
                )
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

        # Stop robot
        data.ctrl[:] = 0

        # =====================================
        # Final evaluation
        # Ground truth is NOT controller input
        # =====================================

        (
            true_x,
            true_y,
            true_heading,
        ) = get_robot_pose(data)

        position_error = math.hypot(
            ekf.x[0] - true_x,
            ekf.x[1] - true_y,
        )

        heading_error = abs(
            wrap_angle(
                ekf.x[2]
                - true_heading
            )
        )

        print(
            "\nFinal EKF:"
            f" x={ekf.x[0]:.3f},"
            f" y={ekf.x[1]:.3f},"
            f" theta="
            f"{math.degrees(ekf.x[2]):.1f} deg"
        )

        print(
            "Ground truth:"
            f" x={true_x:.3f},"
            f" y={true_y:.3f},"
            f" theta="
            f"{math.degrees(true_heading):.1f} deg"
        )

        print(
            "Estimation error:"
            f" position={position_error:.3f} m,"
            f" heading="
            f"{math.degrees(heading_error):.1f} deg"
        )

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
