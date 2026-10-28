#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WORKSPACE_SETUP="${TURTLEBOT3_WS_SETUP:-${SCRIPT_DIR}/turtlebot3_ws/install/setup.bash}"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS 2 setup file not found: ${ROS_SETUP}" >&2
    exit 1
fi

# ROS setup scripts can reference unset variables.
set +u
source "${ROS_SETUP}"
if [[ -f "${WORKSPACE_SETUP}" ]]; then
    source "${WORKSPACE_SETUP}"
else
    echo "Warning: workspace overlay not found: ${WORKSPACE_SETUP}" >&2
fi
set -u

export LDS_MODEL="${LDS_MODEL:-LDS-01}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
export OPENCR_PORT="${OPENCR_PORT:-/dev/ttyACM0}"
export OPENCR_MODEL="${OPENCR_MODEL:-waffle}"
export ROS_LOCALHOST_ONLY=1

pids=()
cleanup() {
    if (( ${#pids[@]} > 0 )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ros2 launch turtlebot3_bringup robot.launch.py &
pids+=("$!")

ros2 run camera_ros camera_node --ros-args \
    -p width:=640 -p height:=480 -p format:=RGB888 &
pids+=("$!")

ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
pids+=("$!")

# If any component exits, stop the others and let systemd restart the stack.
wait -n "${pids[@]}" || true
echo "A robot component exited; stopping the remaining components." >&2
exit 1
