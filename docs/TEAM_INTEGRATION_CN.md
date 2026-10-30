# 团队接入说明

下面是接入模板，不是已经存在的完整导航程序。需要控制/EKF 负责人提供本项目实际变量。
所有输入状态按 `[x_m, y_m, heading_rad]` 排列；测量、估计、参考、真值统一坐标系。

## 1. 在已有主循环检查激光

```python
from lidar_evaluation import ScanData, evaluate_front

# raw_scan 来自已有传感器接口，先保留完整角度序列，包括无效/缺失距离。
scan = ScanData(raw_scan.angles, raw_scan.ranges,
                raw_scan.range_min, raw_scan.range_max)
front = evaluate_front(scan, half_angle_deg=30,
                       stop_distance=0.45, min_valid_fraction=0.5)

# 所有姿态反馈仍来自 EKF。路径规划/绕障由团队导航模块负责。
linear, angular = controller(estimated_state)
if front.status != "clear":
    linear, angular = 0.0, 0.0

# 交给团队已有的执行器接口把 v/omega 换成轮速或力矩，不把 m/s 直接写进 data.ctrl。
send_velocity_command(linear, angular)
```

这是保守停车门限，不是完整避障规划器；检测到障碍后仍需要导航模块产生可通行路线。
0.45 米阈值应根据安装位置、车身尺寸、速度、通信延迟和制动实验确定。
旋转/倒车也需要相应区域检查，前方扇区检查不能证明车身所有方向均安全。
若扫描过期或传感器没有新消息，应输出未知，不沿用过期的 clear 结果。

已有 `rosbridge_api` 的 `query_laser_scan` 可返回接口所需字段。
该 ROS 接口没有保留源时间戳，因此完整系统仍需明确同步和掉线策略。

## 2. 给 MuJoCo 模型添加传感器站点

在实际 LiDAR 的安装位置添加 site。下面仅演示 XML 写法，位置需按实际 TB3 模型标定：

```xml
<site name="lidar_site" pos="0 0 0.2" size="0.005"/>
```

应把 site 放在随车运动的对应 body 里面，不放在静止 worldbody 中。
扫描平面应与车体水平面平行，局部 +X 的安装偏航用 `mount_yaw_rad` 补偿。
将所有车体零件的 geom 设为同一专用 group（例如 1），确保障碍不在此组。

```python
from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

# 初始化时构造一次，不要每个仿真步重复建立传感器对象。
lidar = MujocoPlanarLidar(model, "lidar_site", beam_count=360,
                          range_min=0.05, range_max=8.0,
                          excluded_geom_groups=(1,), mount_yaw_rad=0.0)

# 在主循环的 mj_forward/mj_step 后获取传感器输出。
raw_scan = lidar.read(data)
```

MuJoCo 必须使用自己的位姿和几何计算射线，这是传感器仿真的内部实现。
适配器输出只有角度和距离，不提供真值状态给控制器。
ray casting 模型为瞬时平面扫描，未模拟扫描期间的运动畸变、材质反射或真实硬件电路。

## 3. 同步记录真实闭环试验

每次试验创建一次记录器，`config` 里记录实际使用的参数和初始条件。
下面的数值只是配置示例，不是已完成的实验结果。

```python
from lidar_evaluation.logging import TrialLogger

logger = TrialLogger(
    "results/team_trials", trial_id="trial001", condition="nominal", seed=0,
    config={"source": "actual_team_mujoco_run",
            "initial_pose": [0.0, 0.0, 0.0],
            "lidar_half_angle_deg": 30, "stop_distance_m": 0.45,
            "encoder_noise": "填写实际配置", "camera_dropout": "填写实际配置"},
)

# 每个评估采样点调用：所有量必须取自同一仿真时刻。
# truth_pose 仅流向日志；控制器在其他代码里只读取 estimated_pose。
logger.record_sample(
    time_s=simulation_time,
    truth_pose=truth_pose,
    estimated_pose=estimated_pose,
    reference_pose=reference_pose,
    covariance=ekf_covariance_3x3,
    control=(linear_command, angular_command),
)

# 完成、碰撞或超时后都要结束记录；包含最后一个实际状态。
result = logger.finish(
    goal=goal_pose,
    collision=collision_detected,
    timed_out=timeout_reached,
    position_tolerance_m=0.1,
    heading_tolerance_deg=10.0,
    failure_reason="填写实际观察到的原因或留空",
)
```

不能简单把整个自由关节 `data.qpos` 当作 `[x,y,heading]`；应由队友按正确的坐标系、
四元数顺序提取平面位姿。EKF 若含更多状态，选取 x/y/heading 对应的完整 3×3 子协方差，
包括交叉项。角度和协方差必须都以弧度为单位，绘图时才转换为度。

记录器检查时间递增、输入形状、协方差对称和半正定，但不能自动证明传入的状态来自同一时刻。
要求在开始、运行和终止状态都采样。它在 finish 时写盘，进程崩溃前未 finish 的样本不会落盘；
主程序要在正常终止/超时/受控异常出口调用 finish 并记录真实失败原因。
本记录器用于单进程顺序试验，不支持多进程同时写同一 `trials.csv`。

成功严格由最终位置误差、最短朝向误差、无碰撞、无超时共同计算。
同一条件和编号重复写入会报错，保护旧日志；新试验使用新编号。
条件和编号使用英文字母、数字、单下划线和连字符，不使用路径分隔符或双下划线。

每次试验写出：

- `raw/nominal__trial001.csv`：时间、真值、估计、参考、完整协方差和速度指令。
- `manifests/nominal__trial001.json`：初始条件、噪声参数、目标、容差、失败原因、文件哈希。
- `trials.csv`：每次试验一行，供 report 汇总。

日志在 `results/` 下，已加入 gitignore。发布项目页面时应挑选实际 CSV/图表复制到
例如 `docs/results/` 并提交；不要因为结果目录被忽略就漏交定量证据。

## 4. 这部分尚不能替代的团队工作

- 完整小车模型及可控性/可观性分析。
- 相机内外参标定、ArUco 位姿测量、编码器与视觉的 EKF。
- 用估计状态驱动的 LQR/状态反馈、路径规划、视觉对接。
- 在真实 TB3 场景中的重复任务试验、5 分钟视频和个人答辩。

当前的 `test_forward.py` 仍是团队已有的开环动作测试。本次未修改该文件，
也未把其固定执行器指令当作闭环控制或导航结果。
