# Sensor and Control - Integrated Project


## Project Overview

**Course:** 41014 Sensors and Control in Mechatronics Systems  
**Project:** Track A – TurtleBot3 Visual Waypoint Navigation  
**Platform:** TurtleBot3 Waffle Pi in MuJoCo

This project aims to develop an autonomous mobile robot that can navigate through an environment containing obstacles and dock at an ArUco-marked charging station.

The system integrates wheel encoder odometry, LiDAR sensing, camera-based ArUco detection, Extended Kalman Filter (EKF) localisation, and feedback control.

The controller uses estimated robot states rather than simulator ground truth. Ground truth is used only for performance evaluation.

## System Architecture

The system consists of five main components:

1. **Modelling:** Differential-drive robot kinematics and state-space modelling.
2. **Sensing:** Wheel encoders, 2D LiDAR and an RGB camera.
3. **Estimation:** EKF combining encoder odometry with ArUco-based pose measurements.
4. **Control:** Waypoint tracking using estimated robot states, with state-feedback/LQR development.
5. **Integration:** Obstacle handling, visual docking and robustness evaluation.

### Data Flow

```text
Wheel Encoders ─────┐
                    ├──> EKF ──> Estimated State
Camera + ArUco ─────┘                 |
                                      v
                           Navigation Controller
                                      |
LiDAR ──> Obstacle Detection ─────────┤
                                      |
                                      v
                           Linear / Angular Velocity
                                      |
                                      v
                           Differential Drive
                                      |
                                      v
                              TurtleBot3
```

## Technical Implementation

### A. Robot Modelling

The TurtleBot3 Waffle Pi is modelled as a differential-drive mobile robot.

- Wheel radius: 0.033 m
- Wheel separation: 0.287 m
- Control inputs: linear velocity and angular velocity
- Estimated planar state: [x, y, heading]

State-space modelling and controllability/observability analysis will be documented alongside the final controller.


#### Differential-Drive Kinematics

The TurtleBot3 is modelled as a differential-drive robot with two independently driven wheels.

The robot state is defined as:

$$
\mathbf{x} = [x,\ y,\ \theta]^T
$$

where $x$ and $y$ represent the robot position in the world frame, and $\theta$ represents its heading.

The continuous-time kinematic model is:

$$
\dot{x} = v\cos\theta
$$

$$
\dot{y} = v\sin\theta
$$

$$
\dot{\theta} = \omega
$$

The relationship between body velocities and wheel angular velocities is:

$$
\omega_L = \frac{v-\omega L/2}{r}
$$

$$
\omega_R = \frac{v+\omega L/2}{r}
$$

where:

- $v$: linear velocity (m/s)
- $\omega$: angular velocity (rad/s)
- $r$: wheel radius (0.033 m)
- $L$: wheel separation (0.287 m)
- $\omega_L$, $\omega_R$: left and right wheel angular velocities (rad/s)

These equations are implemented in `main.py` to convert velocity commands into MuJoCo wheel actuator commands.

The nonlinear kinematic model provides the basis for the subsequent state-space and feedback controller design.


### B. Sensing

The simulation includes:

- Wheel joint measurements for encoder odometry.
- A front-mounted RGB camera for ArUco detection.
- OpenCV marker detection and pose estimation.
- A simulated planar LiDAR for obstacle measurement.
- Sensor calibration, noise handling and validation.


#### Camera and ArUco Detection

A front-mounted RGB camera is added to the TurtleBot3 MuJoCo model for visual localisation.

**Camera configuration:**

| Parameter | Value |
|---|---|
| Resolution | 640 × 480 |
| Vertical field of view | 60° |
| Camera update interval | 0.1 s |
| Camera mounting position | (0.10, 0.00, 0.12) m |

The camera captures RGB images directly from the MuJoCo simulation. OpenCV is used to detect ArUco markers and estimate their relative poses.

**ArUco configuration:**

- Dictionary: `DICT_4X4_50`
- Marker ID: `0`
- Marker size: `0.18 m`
- Detection: OpenCV ArUco detector
- Pose estimation: `solvePnP()`

The detected marker corners are converted into relative position and orientation measurements using the camera intrinsic matrix.

The known marker position and camera mounting geometry are then used to estimate the robot's pose in the world frame.

These measurements provide external corrections for the EKF.

#### LiDAR Sensing

A simulated 2D LiDAR is integrated into the TurtleBot3 model using MuJoCo ray casting.

The LiDAR provides range measurements around the robot and is used to identify nearby walls and obstacles.

The sensing pipeline includes:

- Planar laser scanning
- Range validity checks
- Distance calibration and noise evaluation
- Front obstacle detection
- `clear`, `blocked`, and `unknown` status classification

The existing safety logic stops the robot when an obstacle is too close or sensor information is insufficient.

This provides a safety layer for navigation, while active obstacle avoidance is handled by the navigation controller.

#### Coordinate Frames

The project uses consistent coordinate frames for sensing, localisation, and control.

- **World frame:** Global position and heading of the robot.
- **Robot frame:** Attached to the TurtleBot3, with +X forward and +Y left.
- **Camera frame:** Used for ArUco pose estimation through OpenCV.
- **LiDAR frame:** Attached to the robot, with scan angles referenced to the robot's forward direction.

Camera measurements are transformed into the world frame before being used by the EKF.

The controller receives the estimated robot pose:

`[x, y, theta]`

rather than the simulator ground-truth pose.


### C. State Estimation

An Extended Kalman Filter (EKF) estimates robot position and heading.

- Prediction uses wheel encoder odometry.
- Correction uses camera-based ArUco pose measurements.
- Estimated states are passed to the navigation controller.
- Simulator ground truth is retained for evaluation only.

The current implementation includes localisation error reporting and supports comparison under different sensor conditions.


#### EKF Implementation

The Extended Kalman Filter estimates the robot's planar state:

$$
\hat{\mathbf{x}} = [\hat{x},\ \hat{y},\ \hat{\theta}]^T
$$

The EKF combines two sources of information:

**1. Prediction — Wheel Encoder Odometry**

Changes in left and right wheel angles are converted into travelled distance and heading change.

The predicted state is calculated as:

$$
\hat{x}_{k+1} = \hat{x}_k + \Delta d\cos(\hat{\theta}_k+\Delta\theta/2)
$$

$$
\hat{y}_{k+1} = \hat{y}_k + \Delta d\sin(\hat{\theta}_k+\Delta\theta/2)
$$

$$
\hat{\theta}_{k+1} = \hat{\theta}_k+\Delta\theta
$$

The state covariance is propagated using the linearised motion-model Jacobian and process noise covariance Q.

**2. Correction — ArUco Visual Localisation**

The front camera detects the ArUco marker and estimates its relative pose using OpenCV.

The measurement is transformed into the world coordinate frame to obtain an observed robot position and heading.

The EKF corrects its predicted state using the measurement innovation and Kalman gain.

This reduces accumulated wheel odometry drift when valid visual measurements are available.

#### EKF Parameters

The initial implementation uses the following covariance settings:

| Parameter | Value |
|---|---|
| State | [x, y, theta] |
| Initial position variance | 0.01 m² |
| Initial heading variance | (2°)² |
| Process noise Q | diag(0.002, 0.002, (1°)²) |
| Camera measurement noise R | diag((0.03 m)², (0.03 m)², (3°)²) |

These are initial simulation settings. Further tuning and evaluation are required to justify the final noise parameters.

#### Localisation Performance

The implementation compares EKF state estimates with MuJoCo ground truth for evaluation purposes.

An initial standalone test with real ArUco image measurements produced:

- Final position estimation error: **0.020 m**
- Final heading estimation error: **9.1°**

These values represent one preliminary run, not the final performance across repeated trials.

Heading drift remains a limitation when the ArUco marker leaves the camera's field of view, as the filter must rely primarily on wheel odometry.

Final localisation performance will be evaluated using repeated trials, trajectory comparisons and EKF covariance plots.


### D. Control and Navigation

A proportional waypoint controller has been implemented as the initial baseline.

The remaining development includes formal state-feedback/LQR control, improved trajectory tracking, autonomous obstacle avoidance and visual docking.


#### Baseline Waypoint Controller

The initial navigation system uses a proportional controller to move the TurtleBot3 towards a predefined waypoint.

The controller receives the estimated robot pose from the EKF:

$$
\hat{\mathbf{x}} = [\hat{x},\ \hat{y},\ \hat{\theta}]^T
$$

For a target waypoint $(x_t, y_t)$, the distance and heading errors are calculated as:

$$
d = \sqrt{(x_t-\hat{x})^2+(y_t-\hat{y})^2}
$$

$$
e_\theta = \operatorname{wrap}(\operatorname{atan2}(y_t-\hat{y},x_t-\hat{x})-\hat{\theta})
$$

The commanded linear and angular velocities are:

$$
v = \min(K_d d,\ v_{\max})\max(0,\cos(e_\theta))
$$

$$
\omega = \operatorname{clip}(K_\theta e_\theta,-\omega_{\max},\omega_{\max})
$$

The cosine term reduces forward movement when the robot is not facing the target.

**Initial controller parameters:**

| Parameter | Value |
|---|---|
| Distance gain, Kd | 0.8 |
| Heading gain, Kθ | 1.8 |
| Maximum linear velocity | 0.12 m/s |
| Maximum angular velocity | 0.8 rad/s |
| Position tolerance | 0.05 m |
| Initial target waypoint | (1.0, 0.5) m |

The controller stops when the estimated distance to the target falls below the position tolerance.

This proportional controller serves as a baseline for evaluating the final state-feedback/LQR controller.

#### Obstacle Monitoring and Safety

The current integrated implementation uses LiDAR measurements to monitor the robot's surroundings.

In `run_integrated.py`, the safety layer:

- Checks the forward LiDAR scanning region.
- Detects obstacles within a specified stopping distance.
- Monitors missing or outdated sensor measurements.
- Overrides motion commands with zero velocity when conditions are unsafe.

The current integration uses a 0.35 m front-obstacle stopping threshold.

This is an obstacle detection and emergency stopping mechanism, not yet a complete autonomous obstacle avoidance controller.

#### State-Feedback / LQR Controller

The final controller design will be documented after integration, including:

- Linearised state-space model.
- Controller gain or LQR weighting matrices.
- Tracking error formulation.
- Controller tuning and design justification.
- Quantitative comparison with the baseline controller.

#### Visual Servoing and Docking

The final navigation stage is intended to use camera measurements of the ArUco marker to guide the robot towards the charging station.

The final implementation and docking performance will be documented after integration testing.


### E. Integration and Robustness

The integration framework combines localisation, LiDAR obstacle monitoring and robot motion control.

Existing tests include nominal operation, obstacle presence and temporary sensor loss.

Initial testing has identified tracking limitations and localisation drift when the ArUco marker leaves the camera view.

Final performance results will be updated after full navigation and docking integration.


### Integrated System Testing

The integrated simulation combines the TurtleBot3 model, wheel odometry, camera-based ArUco localisation, EKF estimation, waypoint control, and LiDAR obstacle monitoring.

The integration is implemented in `run_integrated.py`, with experimental results recorded for quantitative evaluation.

#### Experimental Conditions

Four conditions were tested:

1. **Nominal:** Normal operation without additional obstacles.
2. **Obstacle:** An obstacle placed along the robot's route.
3. **Camera dropout:** Visual measurements unavailable between 2–5 seconds.
4. **LiDAR dropout:** Laser measurements unavailable between 2–5 seconds.

Each condition was tested three times with different noise seeds and a maximum simulation duration of 45 seconds.

A waypoint trial is considered successful if the robot reaches a ground-truth final position error of 0.05 m or less, without collision or timeout.

#### Preliminary Results

| Condition | Success Rate | Collisions | Mean Final Position Error |
|---|---|---|---|
| Nominal | 0/3 | 0 | 0.154 m |
| Obstacle | 0/3 | 0 | 0.904 m |
| Camera dropout | 0/3 | 0 | 0.157 m |
| LiDAR dropout | 0/3 | 0 | 0.151 m |

These results represent the preliminary integration using the proportional waypoint controller, before the final LQR, obstacle avoidance and docking implementation.

#### Analysis and Limitations

The initial tests identified several limitations:

- **Low-speed motion:** Small velocity commands near the target may be insufficient to overcome simulated wheel friction.
- **Visual localisation:** When the ArUco marker leaves the camera's field of view, EKF correction becomes unavailable and odometry error may accumulate.
- **Obstacle handling:** LiDAR safety stopping works, but the current baseline does not navigate around the obstacle.
- **Tracking accuracy:** The baseline controller did not achieve the required 0.05 m position tolerance in these trials.

No collisions occurred during the recorded trials. However, collision-free operation alone does not demonstrate successful navigation.

#### Further Evaluation

The final integrated controller will be evaluated using the same conditions, including:

- Navigation and docking success rates.
- Final position and heading errors.
- Estimated versus ground-truth trajectories.
- EKF estimation uncertainty.
- Controller performance under sensor dropout and obstacles.

The final results will be compared with the baseline to identify improvements and remaining limitations.

Detailed integration notes are available in [Integration Update](docs/INTEGRATION_UPDATE_CN.md).


## Simulation and Evaluation

The project uses MuJoCo to evaluate robot motion and sensor performance.

Evaluation focuses on:

- Ground-truth versus estimated trajectories.
- Position and heading estimation errors.
- Waypoint tracking performance.
- Obstacle detection and avoidance.
- Robustness under sensor noise or temporary loss.
- Navigation and docking success rates.

Quantitative plots and final trial results will be added after integration testing.


### Running Integration Tests

The integrated simulation can be executed using `run_integrated.py`.

Run a nominal trial:

```powershell
python run_integrated.py --condition nominal
```

Run an obstacle test with the MuJoCo viewer and camera window:

```powershell
python run_integrated.py --condition obstacle --gui
```

Run repeated trials for the four evaluation conditions:

```powershell
python run_integrated.py --condition nominal --trials 3
python run_integrated.py --condition obstacle --trials 3
python run_integrated.py --condition camera_dropout --trials 3
python run_integrated.py --condition lidar_dropout --trials 3
```

These commands use the current baseline waypoint controller and LiDAR safety monitoring. They do not represent the final docking implementation.

### Experimental Data and Outputs

The evaluation system automatically generates CSV files, summary reports and plots.

By default, results are saved under `results/team_integration/`.

| Output | Description |
|---|---|
| `trials.csv` | Trial outcomes, errors and durations |
| `raw/` | Ground-truth, estimated and reference states |
| `sensor_logs/` | LiDAR status and motion commands |
| `plots/` | Trajectory and EKF uncertainty plots |
| `sensor_plots/` | LiDAR response visualisations |
| `report/summary.csv` | Performance summary by condition |

Simulator ground truth is used only for evaluation and does not provide feedback to the navigation controller.

### Visualisation and Results

The project evaluation includes three main types of visual evidence:

**1. Simulation Environment**

MuJoCo visualisation of the TurtleBot3, surrounding environment, obstacles and ArUco marker.

**2. Robot Trajectory and Localisation**

Comparison between ground-truth position, EKF estimated position and reference waypoint.

**3. Estimation Uncertainty and Sensor Response**

EKF estimation errors and covariance bounds, together with LiDAR obstacle detection and safety-stop behaviour.

Representative figures from the final tests will be included in this section before submission.

### Reproducibility

The simulation uses fixed initial conditions and recorded noise seeds for comparison between test conditions.

Each trial records its experimental configuration and results. Repeated tests are used to evaluate performance rather than relying on a single demonstration.

Final results will be updated after the complete navigation controller and visual docking system are integrated.


## Installation and Running

The repository includes the modified TurtleBot3 MuJoCo model, camera configuration and ArUco marker assets under `models/robotis_tb3/`.

From the repository root, install the required Python dependencies and run the main simulation:

```powershell
python -m pip install -e ".[evaluation,simulation]"
python main.py
```

For the LiDAR processing and evaluation module:

```powershell
python run_my_part.py
```

Additional integration scripts and testing instructions are documented in the project files.

## Demo Video

The final five-minute demonstration will present the system architecture, sensing, estimation, control, navigation and robustness results.

**Video link:** To be added before submission.


## Team Contributions

The project was developed collaboratively by three team members, with responsibilities divided across simulation, sensing, estimation, control and evaluation.

| Team Member | Main Contributions |
|---|---|
| [Member 1] | MuJoCo simulation setup, TurtleBot3 model integration, differential-drive kinematics, wheel encoder odometry, RGB camera and ArUco detection, EKF localisation and initial waypoint control. |
| [Member 2] | LiDAR simulation and processing, range calibration, obstacle detection and safety monitoring, experimental logging, repeated trials and performance evaluation. |
| [Member 3] | State-space modelling, controllability and observability analysis, state-feedback/LQR controller, navigation and obstacle avoidance, and visual docking implementation. |

All members contributed to project integration, troubleshooting and preparation of the final demonstration and documentation.

## Generative AI Declaration

Generative AI tools were used to assist with project development and documentation.

**Tools used:**
- **OpenAI ChatGPT:** Technical explanations, implementation guidance, troubleshooting and assistance with organising project documentation.
- **OpenAI Codex:** Assistance with code development, debugging, integration testing and technical documentation.

AI-generated suggestions and code were reviewed and tested by team members before being incorporated into the project.

The team remains responsible for the final implementation, experimental results, technical decisions and submitted work.



## References and Acknowledgements

The project builds upon course tutorial material, MuJoCo examples and the ROBOTIS TurtleBot3 model resources. Adapted material and external resources will be acknowledged in the final project documentation.

---



## LiDAR processing and evaluation contribution

Chinese implementation and viva guide: [docs/MY_PART_CN.md](docs/MY_PART_CN.md).
Team integration example: [docs/TEAM_INTEGRATION_CN.md](docs/TEAM_INTEGRATION_CN.md).

```powershell
python -m pip install -e ".[evaluation,simulation]"
python run_my_part.py
```

This runs explicitly labelled sensor-component checks and writes CSV/JSON/PNG files
to `results/my_part/`. It does **not** claim TurtleBot3 navigation or docking success.
The report and logging modules accept the team's actual closed-loop trial data.
The new contribution includes Chinese comments. Existing interface documentation follows.

## Student Rosbridge API

A small Python API for students who use an ML model or robotics modelling
toolbox to control a ROS robot through rosbridge. It exposes only the main
operations a student control loop needs:

- sign in to borrow the robot and another sign-in for the battery   
- log into `turtlebot-MMR wifi`, pwd: `Turtlebot`
- ping 192.168.1.2XX where XX is the TurtleBot ID   
- query the latest compressed camera image as a BGR NumPy array;
- query the latest laser scan as angle and range NumPy arrays;
- publish linear and angular velocity to `/cmd_vel`;
- stop and disconnect safely.

## Robot requirements

The robot must:

1. run a rosbridge WebSocket server reachable by the student's computer;
2. accept `geometry_msgs/msg/Twist` messages on `/cmd_vel`;
3. publish `sensor_msgs/msg/CompressedImage` messages on
   `/camera/image_raw/compressed`;
4. publish `sensor_msgs/msg/LaserScan` messages on `/scan`.

Topic names can be changed when constructing `RosbridgeRobot`.

## Install locally for testing

From this directory:

```bash
python -m pip install .
```

Include the keyboard teleoperation dependency with:

```bash
python -m pip install ".[teleop]"
```

For active development, use an editable installation:

```bash
python -m pip install -e ".[teleop,dev]"
```

## ML/control-loop example

```python
from rosbridge_api import RosbridgeRobot

with RosbridgeRobot("192.168.1.213") as robot:
    while True:
        image = robot.query_image(timeout=2.0, wait_for_new=True)
        if image is None:
            continue

        linear_velocity, angular_velocity = controller(image)
        robot.send_command(linear_velocity, angular_velocity)
```

`query_image()` returns a copy, so student code may safely annotate or resize it.
Its colour order is BGR, as expected by OpenCV.

## Included examples

The runnable examples are kept separately from the library in `examples/`.
After installing the package, run them from the project directory:

```bash
python examples/camera.py 192.168.1.213
python examples/laser_scan.py 192.168.1.213
python examples/teleop.py 192.168.1.213
```

The camera and laser-scan visualisers exit with `Q` or Escape. The teleop
controls are W/S for forward/backward, A/D for turning, Space for an immediate
stop, and Escape to exit.

## Test and build

```bash
python -m unittest discover -s tests -v
python -m build
python -m twine check --strict dist/*
```

The tests mock rosbridge and do not require a physical robot.

Before publishing, choose the final project name, add author/project metadata,
choose a licence, and verify installation from the generated wheel in a clean
virtual environment.
