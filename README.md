# Sensor and Control - Integrated Project

## Updated team integration

The latest team model, camera/ArUco localisation and EKF are now connected to the
LiDAR gate and trial logger through `run_integrated.py`.
See [updated Chinese integration report](docs/INTEGRATION_UPDATE_CN.md) for scope,
commands, measured failures and the remaining controller/docking work.

```powershell
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition obstacle --gui
```

Obstacle handling currently stops the robot; it does not plan an avoidance route.
The original waypoint controller is temporary and is not an LQR/docking controller.

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
