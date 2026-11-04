# 完整仿真、运行方法和中文答辩说明

## 现在能完成什么

新的入口是 `run_navigation.py`。旧的 `run_integrated.py` 保留为临时比例控制基线，
不要继续用它判断最终导航是否完成。

新系统从真实 LiDAR 扫描建立占据图，用 A* 规划路径，再用基于估计状态的 LQR 跟踪。
接近充电站中心线后搜索 ArUco，使用标定后的完整相机坐标变换与 EKF，进行 PBVS 位姿对接。
必要时先后退调整接近路线，避免差速小车无法横移而在近距离原地转出相机视野。
只有位置、朝向、最近视觉观测和持续稳定条件都满足，才进入 DOCKED。

充电站是 ArUco 背板与蓝色地面停靠区；对接后停靠区变绿。
这里模拟几何位姿停靠，不模拟实际充电电流或电气接触。

## 在 VS Code 运行

先关闭旧窗口。在 Sensor-and-Control 根目录执行：

```powershell
.\.venv-lidar\Scripts\python.exe run_navigation.py --gui --speed 2
```

默认带红色障碍。出现两个窗口：MuJoCo 房间与相机视图。相机会显示阶段、仿真时间和激光状态。
完成后会保持最终画面约2.5秒再关闭，并生成真实日志和图表。

阶段含义：

| 阶段 | 含义 |
|---|---|
| NAVIGATE | 激光建图、规划与 LQR 跟踪 |
| ACQUIRE_MARKER | 原地朝向已知站点方位，等待真实图像测量 |
| ALIGN_APPROACH | 先退开调整接近路线，后方也要通过激光车体检查 |
| VISUAL_DOCK | 基于视觉修正后的估计位姿进行精细对接 |
| DOCKED | 估计和视觉确认稳定，运动指令清零 |

`--speed 2` 只改变 GUI 播放节拍，不改变物理步长、控制器或车轮速度指令。
有障碍的种子通常需要40到75秒仿真时间，包含寻找标记和调整路线；不是卡死时可以继续等待。
存在接近路线重复调整，控制还不是时间最优。

其他命令：

```powershell
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition nominal --gui
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition obstacle --trials 3 --duration 100
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition camera_dropout --trials 3 --duration 100
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition lidar_dropout --trials 3 --duration 100
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition noisy_encoder --trials 3 --duration 100
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition shifted_start --trials 3 --duration 100
.\.venv-lidar\Scripts\python.exe run_navigation.py --condition obstacle --seed 2 --record-video
.\.venv-lidar\Scripts\python.exe calibrate_navigation.py
```

新电脑先安装 `python -m pip install -e ".[evaluation,simulation]"`，建议 Python 3.12。
本机已配置的 `.venv-lidar` 可以继续使用。完整功能使用 MuJoCo MjSpec，已测试版本为3.17.0。

## 模型与控制设计（对应 A、D）

状态为 `[x,y,theta]`，输入为 `[v,omega]`。

\[
\dot x=v\cos\theta,\quad \dot y=v\sin\theta,\quad\dot\theta=\omega.
\]

轮半径采用0.033米，名义有效轮距0.287米。XML两轮中心间隔约0.288米，名义参数的微小差异、
轮胎滑移与地面接触被作为模型失配，不能当作无误差运动学。反馈控制和视觉修正用于抵抗这些误差。

在沿参考方向移动、名义速度0.12m/s、dt=0.02s时，误差为沿程、横向和朝向误差：

\[
e_{k+1}=A_de_k+B_d\delta u_k,
\quad A_d=\begin{bmatrix}1&0&0\\0&1&0.0024\\0&0&1\end{bmatrix},
\quad B_d=\begin{bmatrix}0.02&0\\0&0\\0&0.02\end{bmatrix}.
\]

`[B,AB,A²B]` 的秩为3；视觉位姿测量 H=I，可观性矩阵的秩为3。
静止参考速度为0时，同一线性化的可控性秩降为2。这不意味着差速车全局不可达，
只意味着不能用静止点的这个局部线性模型直接证明横向可控。原地转向和最终对接采用单独的非线性反馈。
编码器被当作运动预测输入，单独不能观测世界坐标下累计位姿漂移。

离散 LQR 使用 Q=diag(6,40,10)、R=diag(4,2)，数值增益约为：

\[
K=\begin{bmatrix}1.20984&0&0\\0&4.36328&2.41496\end{bmatrix}.
\]

闭环特征值约0.95652、0.99518、0.97580，均位于单位圆内。
增大横向误差权重有利于保留障碍膨胀后的间距；输入权重限制过激转向。
这是有依据的初始设计，不是全局最优或任意场景稳定性证明。
跟踪器的名义前馈约0.10m/s，并叠加误差反馈，最终限幅到0.14m/s及0.75rad/s。
路径是几何路径，参考点随估计位置推进；日志里的tracking RMSE是这种在线几何参考的误差，
不是提前给定时间表的轨迹跟踪指标。

原模型关节 frictionloss=0.1、速度伺服kv=0.1，存在明显低速死区。
新轮速内环保留原物理参数，使用编码器反馈PI与等效1rad/s的摩擦前馈。
增益为0.45比例、0.8积分，积分限制±2，执行器命令限制±7.88。
停止时清零命令和积分，避免为了演示而删除摩擦。

## 传感、坐标系和标定（对应 B）

- 车体 +X 向前、+Y 向左；角度为弧度、距离为米。
- 相机为640×480，fovy=60度，虚拟针孔相机无畸变。
- 内参由虚拟相机配置推导并通过12个独立已知位姿验证，fx=fy≈415.692。
- ArUco 字典4X4_50、ID0、黑色标记实际尺寸0.18米，已知世界位置(1.5,0,0.2)。
- 相机安装位置(0.10,0,0.12)，完整旋转从模型安装四元数取得，并转换OpenGL/OpenCV约定。
- LiDAR安装位置(-0.049,0,0.2)，180束，最大量程5米；车体全部geom被排除。

完整变换为 `T_WB = T_WM * inverse(T_CM) * inverse(T_BC)`。
相机通过实际渲染图像检测角点，使用亚像素角点修正与ITERATIVE PnP。
重投影误差大于2像素、不合理深度或姿态的解会被拒绝，之后还经过 EKF 创新门控。
不能使用仿真器真实车体位姿来挑选视觉测量。

虚拟相机已知位姿验证的位置RMSE约0.00576米、朝向RMSE约0.243度。
这是静态、理想虚拟相机验证，不是动态全任务定位误差或真实硬件标定精度。
理想射线量程几何误差接近浮点精度，同样不能解释为真实雷达精度。
早期有人工注入偏差的测距校正实验保留在原文档，来源与本次真实模型验证分开标注。

## 估计（对应 C）

50Hz读取轮关节编码器，10Hz读取相机。预测采用中点差速运动模型。
过程噪声由轮增量噪声传播，加上位置0.02m/√s、朝向3deg/√s的模型失配项。
这些是针对滑移与当前仿真试验选择的参数，不宣称由实物测量得到。
视觉R=diag(0.025²,0.025²,(3deg)²)。
创新马氏距离平方门限16.27用于拒绝不一致测量；Joseph形式更新协方差，保持数值对称半正定。
相机丢失时只用里程计预测，不伪造图像测量。激光丢失时停止运动，数据恢复后重新规划。

## 绕障与视觉对接（对应 E）

占据图由扫描与估计位姿实时建立，不向规划器传递障碍真实坐标。
地图分辨率0.07米，障碍膨胀约0.24米并包含栅格裕量，A*禁止穿过对角障碍角点。
未知栅格代价较高，前进还必须经过原始激光的短期弧线和车身半径检查。

最终目标为车体位姿(1.05,0,0)，与标记平面保持约0.45米车体参考点距离。
PBVS通过相机校正后的估计位姿计算位置和朝向误差；原地调整、退开、重新接近均受激光检查。
估计位置误差<2厘米、估计朝向<2.5度，最近视觉位置误差<2.5厘米、朝向<3度，
连续保持0.6秒后进入DOCKED。独立真值验收必须位置≤5厘米、朝向≤5度、无碰撞、无超时。
容差是本项目明确采用的标准，并非老师统一规定值。

## 当前实测范围和限制

结果表和全部完整试验位于 [results/RESULTS_TABLE.md](results/RESULTS_TABLE.md)。
六种条件各3个种子，完成的18次试验均成功、无碰撞。非nominal条件带有轻微随机位置变化的障碍。
shifted_start 使用已知偏移起点先验；不是在未知初始位姿下自动全局定位。
一次多试验执行在 shifted_start 种子2完成前中断，同一种子重跑通过；中断说明和部分传感器日志保留。
不能把18个完整结果解释为所有尝试都没有执行故障，也不能保证任意房间、动态人群或硬件效果。

受相机视野限制，绕障后可能多次重新搜索标记和调整接近路线，耗时较长。
目前只评估简单静态房间、一个矩形障碍、小范围位置变化、固定地标与光照。
没有SLAM、动态障碍预测或真实电气充电。仿真激光也没有完整模拟反射材质和扫描畸变。
历史临时比例控制的12次失败结果保留。新版本同时更改规划、视觉变换、EKF和内环，
因此新旧结果属于整体系统比较，不能把所有改善单独归因于LQR。

## 文件和演示

新代码在 `src/navigation/`，保留队友原始 `main.py`、旧联调和你的早期LiDAR模块。
项目入口 `run_navigation.py`；校准检查 `calibrate_navigation.py`。
模型数值证据 `docs/results/model_analysis.json`；原始CSV/配置/图表在 `docs/results/navigation/`。
演示视频在 `docs/demo/`，直接记录实际物理仿真、真实相机和估计/真值/不确定性，没有关键帧伪造运动。

正式提交前仍须由团队填写组号、真实成员信息和个人贡献，审查并上传GitHub及视频链接到Canvas，
并按课程安排个人答辩。答辩无法由代码替你完成。

## 简单英语答辩

**What does the robot do?**
It uses LiDAR to navigate around an obstacle. Then it uses the ArUco marker to dock.

**Which state does the controller use?**
It uses the estimated state. Ground truth is used only for evaluation.

**Why was the robot slow before?**
The wheel model has friction. We added encoder feedback and friction compensation.

**What does the EKF combine?**
It combines wheel odometry and camera pose measurements.

**What happens when LiDAR is missing?**
The robot stops. It continues after valid measurements return.

**Why does the robot sometimes move backwards?**
It cannot move sideways. It backs away to get a better approach while checking the clearance.

**How do you decide that docking is complete?**
The estimated pose and a recent camera measurement must stay close to the target. We then check the true error independently.

**What are the limitations?**
We tested a simple static room. The robot may need several attempts to keep the marker in view.

AI声明必须如实包括：Codex辅助实现规划、控制、估计、联调、注释、测试、文档和演示生成。
团队应理解和审查这些实现，贡献声明只记录各成员实际承担的工作，不能把生成代码直接声称为独立手写。

## 重建字幕演示

以下命令生成与当前视频相同条件的三段实际录像，再编排证据卡片：

```powershell
python run_navigation.py --condition nominal --seed 0 --run-id demo4 --record-video --output-dir results/demo-visible
python run_navigation.py --condition obstacle --seed 2 --run-id demo5 --record-video --output-dir results/demo-visible
python run_navigation.py --condition lidar_dropout --seed 2 --run-id demo6 --record-video --output-dir results/demo-visible
python tools/build_demo.py
```

已有同编号文件时先选择新的输出目录或按自己的文件名调整编排脚本；不要覆盖原始证据。
