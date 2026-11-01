# 最新团队代码与我的联调工作

## 队友完成了什么

已合入 main 提交 `98fd60e`，它包含你的早期提交 `5b07f59`，以及队友的两次更新：

- `68cc3a7`：main.py、差速轮速换算、编码器里程计、相机图像、ArUco 位姿测量和 EKF。
- `98fd60e`：仓库内 TB3 模型、前置相机 XML、ArUco 图像、STL 等资源。

相机/EKF 的代码存在且可以运行，但不能因此直接断言定位、控制或完整任务达标。
原 main.py 的航点控制器是距离/朝向比例控制，不是 LQR，也没有最终充电站朝向对接。

## 本次完成的个人工作

1. 保留已有 LiDAR/校准/评估代码，把最新 main 快进合入 lidar-evaluation 分支。
2. 在实际 TB3 上添加随车运动的 LiDAR site；按车体 body 子树排除全部自身几何。
3. 在内存中添加四面墙和可选红色障碍，不改队友模型资源。
4. 用实际射线测距执行 clear/blocked/unknown 检查；近障碍或数据缺失时速度指令清零。
5. 把真实渲染的 ArUco 测量送入队友 EKF，用其估计状态调用原临时控制器。
6. 记录同步真值/估计/参考、协方差、速度、激光状态和真实试验结果。
7. 修复 test_forward.py 的模型路径，增加无界面验证；在 main.py 中做最小 PnP 兼容性修复。
8. 增加中文注释、真实模型回归测试、实际闭环重复试验和失败诊断。

真实位姿只在评价和日志中读取。相机使用实际渲染图像，没有把真值加噪声冒充视觉测量。
噪声与丢失条件是人为设置的仿真实验条件，配置与种子记录在日志。

## 发现并修复的问题

- 原 test_forward.py 仍指向外部模型目录；现在使用仓库自带 `models/robotis_tb3`。
- 当前 OpenCV 5.0 环境的 IPPE_SQUARE 在近正视的初始图像上给出了错误平面解：
  初始零位姿被估成约 x=3.01米、朝向179度。改用 ITERATIVE 后同一图像约 x=-0.012米、朝向0度。
  这是本机复现的兼容性问题，不是“所有 IPPE 算法都错误”的结论。
- 原场景没有房间边界；大部分激光无回波。保守算法正确输出 unknown，但小车因此不能前进。
  添加实际墙面后可获得有效回波，不把 Inf 强行当作无障碍。
- 车体 geom 分散在多个 group；现在按 body 子树统一排除，避免自测距。

## 实测结果与结论

前进测试：固定轮速指令运行8.002秒，实际向 +X 前进约0.4512米。

联调条件：正常、红色障碍、相机2到5秒丢失、LiDAR2到5秒丢失。
每种条件3个种子，每次最长45秒，使用相同的队友临时控制器和相同初始位姿。
只有噪声种子不同，尚未覆盖不同起点、复杂布局或大规模鲁棒性测试。

| 条件 | 航点成功/试验数 | 碰撞次数 | 最终位置误差均值（约） |
|---|---:|---:|---:|
| nominal | 0/3 | 0 | 0.154米 |
| obstacle | 0/3 | 0 | 0.904米 |
| camera_dropout | 0/3 | 0 | 0.157米 |
| lidar_dropout | 0/3 | 0 | 0.151米 |

成功要求真实最终位置误差≤0.05米、没有碰撞且没有超时。
因临时控制器未控制最终朝向，日志明确设置 heading_tolerance_deg=180，
这里只评价“航点位置”，绝不宣称充电站 docking 成功。

障碍试验中每次都触发 blocked；LiDAR 丢失期间触发 unknown。
所有记录的非 clear 样本，其线速度和角速度指令均为零，没有停车指令违规。
这是传感器和停车门限正确接入的证据，不是完整避障成功证据。

正常条件下也未达到5厘米容差。末段指令较小而轮子几乎不动，
模型轮关节存在 frictionloss，临时比例控制可能存在静摩擦死区；需要控制负责人进一步验证和补偿。
同时，拐向目标后标记会离开视野，EKF 依赖里程计，误差与协方差增大。
这两点应共同诊断，不能仅凭最终误差把原因全部归结为控制器或 EKF。

轨迹参考是固定目标点，因此汇总中的 tracking_rmse_m 表示到该固定航点的距离 RMSE，
不是沿时间参数化参考轨迹的跟踪 RMSE。以后接入正式轨迹后应记录对应时刻的参考。

## VS Code 运行命令

在项目根目录执行，无需再次激活环境：

```powershell
.\.venv-lidar\Scripts\python.exe test_forward.py --headless
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition obstacle --gui
```

第二条显示 MuJoCo 和相机窗口，红色障碍在机器人通往航点的方向。它会停车，不会自行绕开。
窗口关闭或按Q/ESC会记录未完成试验。GUI入口存在，自动验证使用无界面模式，不宣称已人工检查完整演示视频。

无界面重复实验：

```powershell
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition nominal --trials 3
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition obstacle --trials 3
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition camera_dropout --trials 3
.\.venv-lidar\Scripts\python.exe run_integrated.py --condition lidar_dropout --trials 3
```

自动生成唯一运行编号；需要复现配置时可指定新的 `--run-id`。重复编号拒绝覆盖旧日志。
默认目录 `results/team_integration`：raw、sensor_logs、manifests、plots、sensor_plots、report。
正式重复实验保存为 `results/integrated_evaluation`，失败和原始数据全部保留。
日志器在 finish 时落盘；不可恢复的进程崩溃可能丢失当前试验，已有试验不受影响。

## 下一步分工

你：把这些实际结果交给负责控制的组员，解释激光状态、停车和误差图；正式控制器更新后复测相同条件。

控制负责人：状态空间模型、可控性/可观性、LQR或正式状态反馈、低速执行器问题、绕障路线和最终对接。

定位负责人：确认相机坐标/标定，处理离开视野后的漂移，解释 EKF 的 Q/R 与更新频率。

三人共同：跑通完整任务、整理定量证据、5分钟视频、个人贡献和AI声明，再提交。
本次代码和说明继续由 Codex 辅助实现，使用情况须在项目页面如实声明。
