# 我的部分：LiDAR 数据处理、距离校正和实验评估

## 1. 范围与运行

你的技术贡献是传感器处理、测距验证、实验日志和结果分析。代码包含中文说明。
目前团队仓库尚未提供完整的 TB3 相机/EKF/LQR 闭环；本模块已经具备独立验证和接入接口，
但仍须与队友系统联调，才能报告整车导航、避障和视觉对接结果。

在 VS Code 打开 `Sensor-and-Control` 项目。安装依赖后，在项目根目录运行：

```powershell
python -m pip install -r requirements-lidar.txt
python run_my_part.py
```

如果电脑的 `python` 是 Windows 商店占位程序，使用已经安装的 Python 3.12：

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv-lidar
& ".\.venv-lidar\Scripts\python.exe" -m pip install -r requirements-lidar.txt
& ".\.venv-lidar\Scripts\python.exe" run_my_part.py
```

无需激活虚拟环境，也无需修改 PowerShell 执行策略。若已经由 Codex 配好了 `.venv-lidar`，
只运行最后一行。双击打开代码并不等于执行代码；所有命令都在终端中运行。

默认同时执行合成逻辑自检和 MuJoCo 静态传感器检查，不弹出机器人动画窗口。
这是无界面的数值验证，结果通过图表呈现。不需要下载 TB3 模型。

## 2. 各文件做什么

| 文件 | 责任 |
|---|---|
| `src/lidar_evaluation/lidar.py` | 统一扫描格式、检查前方区域、输出 clear/blocked/unknown |
| `calibration.py` | 拟合线性测距校正，计算 MAE、RMSE、bias、std |
| `mujoco_scan.py` | 从 MuJoCo site 发射射线，生成角度和距离 |
| `demo.py` | 可复现的合成自检和 MuJoCo 传感器实验 |
| `metrics.py` | 读取实际试验 CSV，计算轨迹误差和条件统计 |
| `logging.py` | 记录真实运行的同步轨迹、EKF 协方差、配置和成功条件 |
| `report.py` | 生成实际试验汇总 CSV/JSON 和成功率/误差图 |
| `plotting.py` | 画真实/估计/参考轨迹、估计误差与边际正负两倍标准差 |
| `run_my_part.py` | 统一运行入口 |
| `tests/test_*.py` | 已知答案、无效数据、坐标变换、缺失数据与日志保护测试 |

## 3. 激光处理逻辑

输入 `angles` 的单位是弧度，`ranges` 的单位是米。车体 +X 是前方，+Y 是左方。
示范前方区域为左右各 30 度；停车距离 0.45 米、最低有效光束比例 0.5。
这些只是演示配置，不是针对实际 TurtleBot3 验证过的安全参数。

1. 把 0..2π 和 -π..π 的角度统一。
2. 在预期的前方光束中检查有效量程。丢失光束保留角度，距离设 NaN/Inf。
3. 有效量程内出现近障碍时输出 `blocked`。
4. 前方出现低于最小量程的非负读数，或有效光束不足时输出 `unknown`。
5. 只有有效覆盖足够且未见阈值内障碍时输出 `clear`。

无回波 Inf 在不同驱动中可能含义不同，本实现保守地当作未知。
`safe_to_move` 只表示这次前方扇区检查通过，不保证整个车身轨迹安全。
没有同步新扫描、扫描过期或传感器掉线时，调用方也必须输出未知并减速/停车。
目前原 ROS 接口未保留传感器时间戳，不能用本机接收时间证明传感器同步。

安装角度不一致时，需要用标定得到的安装偏航角转换光束角度。
MuJoCo 适配器支持 `mount_yaw_rad`，扫描平面需与车体水平面平行。
所有车体 geom 应分配独立 group，并从激光射线中排除，避免激光测到自身。

## 4. 怎样理解校准和现有结果

校正形式为：`corrected = slope * measured + intercept`。
使用校准集拟合参数，使用另一批独立采样的验证集计算校正前后误差。
误差符号为 `measured - reference`，std 使用总体标准差 ddof=0。

默认演示在 0.3、0.6、1.0、1.5、2.0 米的几何参考距离上检查传感器。
人为注入比例误差 1.02、固定偏差 0.04 米、标准差 0.01 米的高斯噪声。
这是一项算法验证，并非实测得出的真实雷达参数。
训练集 60 点，独立验证集 40 点；验证距离与训练距离相同，所以不支持对未见距离泛化的结论。

MuJoCo 扫描检测有 4 个条件，每条件 20 个种子：正常、近障碍、20% 光束丢失、全部缺失。
扫描噪声标准差为人为设置的 0.003 米。输出中的 80/80 是传感器状态检查的通过次数，
绝不能写成 80 次 TurtleBot3 导航全部成功。

`results/my_part/self_check/`：纯合成逻辑验证。

`results/my_part/mujoco_check/`：真实 MuJoCo 射线产生的静态传感器验证，其中误差/丢失为人为注入。

每个目录含 `component_checks.csv`、`calibration_samples.csv`、`calibration_result.json`、
`run_summary.json`、`component_checks.png`、`calibration_validation.png`。
MuJoCo 目录还含 `geometric_range_validation.csv`，直接公开理想测距与已知几何的差异。
图中的英文标题方便放入英文项目页面；源代码的注释用中文解释。

## 5. 与队友一起完成的实际试验

按 [接入说明](TEAM_INTEGRATION_CN.md) 让队友在主循环调用传感器处理和日志记录。
建议先约定成功条件，例如位置误差≤0.1米、朝向误差≤10度、未碰撞且未超时。
这些数值是起点，应由团队按任务要求验证并在 manifest 中固定记录。

对正常、不同起点、增大编码器噪声、相机短时丢失、扰动等条件重复试验。
记录每次种子、初始条件、配置、真值、估计、参考和 EKF 协方差。
失败也必须写入日志，不能只留下成功的视频。
估计和真值必须同一时刻、同一坐标系、同一单位；不能只截断数组长度来假装同步。

已有真实 `results/team_trials/trials.csv` 后运行：

```powershell
python run_my_part.py --mode report --trials-csv results/team_trials/trials.csv
```

只有表头的 `data/trials_template.csv` 是空模板，不是实验数据。
程序会拒绝把空模板当作结果。缺失误差留空，不当成 0；成功率和耗时均包含失败试验。
误差图只包含实际有记录的样本，应同时报告对应样本数和可能的缺失偏差。

如果要画单次轨迹及不确定性，把命令里的文件名改成实际日志：

```powershell
python run_my_part.py --mode plot-trial --trajectory-csv results/team_trials/raw/nominal__trial001.csv
```

这里使用真值仅为了评价，控制器仍只能使用 EKF 估计状态。

## 6. 你必须能解释的简单答辩内容

下面是完成并理解相应工作后可用的短句，按你实际完成的内容调整。

**What was your contribution?**

I worked on LiDAR processing, sensor validation, and performance evaluation.

我负责激光处理、传感器验证和性能评估。

**How do you detect an obstacle?**

I check the valid measurements in the front sector. If an obstacle is too close, the status is blocked.

我检查前方区域的有效测量；近障碍使状态变成 blocked。

**What happens when the data is missing?**

The status becomes unknown. Missing data does not mean the path is clear.

数据不足会变为 unknown；缺失数据不能说明道路畅通。

**How did you check calibration?**

I fitted the correction on one dataset and checked the errors on a separate dataset.

我用一组数据拟合，用另一组数据检查误差。

**Are these navigation results?**

The included demo checks the sensor component. Navigation results need the complete team pipeline.

现有演示验证传感器组件；导航结果需要团队完整系统。

**Why do you use ground truth?**

Ground truth is used only for evaluation. The controller uses the estimated state.

真值只用于评价；控制器使用估计状态。

还要理解其他模块：编码器用于预测、视觉修正漂移、EKF 融合信息、控制器根据估计误差产生速度。
每人答辩会涉及自己没有主导的部分，不能只记住上面的句子。

## 7. 来源、贡献与 AI 声明

原仓库 ROS 接口、示例和前进测试属于团队已有/课程起始代码；本次添加的 lidar_evaluation、
测试、运行入口和中文说明应与继承代码分别说明，不把继承提交者数量当作个人工作证据。

AI 声明示例（按实际情况修改，不要声称已经做了尚未做的工作）：

> OpenAI Codex was used to assist with LiDAR processing code, evaluation tools, Chinese comments,
> tests, and documentation. The team is responsible for reviewing, understanding, integrating,
> and validating the submitted work. Component demos use explicitly labelled simulated data.

参考接口文档：

- MuJoCo ray casting: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-ray
- ROS LaserScan: https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/LaserScan.msg

未自动上传 GitHub，未替你发送组员消息。理解并联调完成后再记录真实个人贡献。
