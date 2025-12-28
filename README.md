# 🌲 EcoGuardian - 智能森林消防多智能体仿真系统

**EcoGuardian** 是一个基于 Python (Pygame + NumPy) 开发的高性能多智能体森林消防仿真平台。该系统模拟了复杂的火灾蔓延动力学，并集成 **UGV (地面机器人)**、**UAV (空中无人机)** 和 **SupportBot (补给机器人)** 三种异构智能体，通过 **在线机器学习 (Online ML)**、**遗传算法 (GA)** 和 **自适应 A\* 路径规划** 实现高效的分布式协同灭火。

---

## ✨ 核心特性 (Key Features)

### 1. 🔥 高保真物理环境 (Physics Engine)

- **向量化元胞自动机**：基于 NumPy 的矩阵运算实现高性能并行仿真，支持 60FPS 流畅运行。
- **多维环境拟合**：包含动态风场、干燥度累积、燃料衰减及 **自燃 (Spontaneous Ignition)** 机制。
- **稀疏矩阵优化**：利用 `np.argwhere` 稀疏索引加速火势蔓延计算，避免全图遍历。

### 2. 🤖 异构多智能体协同 (Heterogeneous Agents)

- **🚁 侦察无人机 (UAV)**：采用 **蒙特卡洛随机采样** 策略，在高空执行 广域扫描，构建全局感知“战争迷雾”。
- **🚜 灭火机器人 (UGV)**：
- 基于 **FSM (有限状态机)** 管理 IDLE, MOVING, RETURNING, STRANDED 状态。
- 具备 **3x3 AOE 强力灭火** 能力及 **连击清理 (Mopping Up)** 策略。
- 集成 **Logistic Regression** 模型，基于电量、水量、风向对齐度等 **6D 特征** 进行任务竞价。

- **🚑 补给机器人 (SupportBot)**：
- 全图监控友军状态，一旦发现搁浅 (Stranded) 立即触发救援。
- 拥有 **2 倍速移动** 能力与 **强力推土机模式** (自动清除路径上的火障)。

### 3. 🧠 混合智能决策架构 (Hybrid AI)

- **微观决策 (Online Learning)**：UGV 内置逻辑回归预测器，具备 **启发式初始化 (Heuristic Init)**、**学习率衰减** 及 **权重剪裁** 机制，能在仿真运行中实时修正决策权重。
- **宏观优化 (Genetic Algorithm)**：后台运行遗传算法，针对全局参数（如竞价惩罚系数、避嫌半径）进行演化。采用 **精英保留 (Elitism)** 与 **高频变异 (Rate=0.5)** 策略，动态适应环境变化。
- **自适应路径规划 (Adaptive A\*)**：
- 实现 **延迟删除 (Lazy Deletion)** 机制优化优先队列性能。
- 引入 **动态火场代价 (Dynamic Fire Cost)**：持水时视火为通路 (Cost=1)，无水时视火为障碍 (Cost=50)。

---

## 📂 项目结构 (Structure)

```text
EcoGuardian/
├── main.py                  # 程序入口与主循环
├── configs/
│   └── settings.py          # 全局配置 (颜色、地图尺寸、物理参数)
├── core/
│   ├── grid_map.py          # 物理引擎 (火势蔓延、干燥度、自燃)
│   ├── genetic_optimizer.py # 遗传算法优化器 (GA)
│   └── pathfinding.py       # 路径规划算法 (A*)
└── agents/
    ├── base_agent.py        # 智能体基类
    ├── robot.py             # 地面机器人 (UGV) 与 补给机器人 (SupportBot)
    ├── drone.py             # 无人机 (UAV)
    └── predictor.py         # 机器学习预测器 (ML)

```

---

## 🚀 快速开始 (Quick Start)

### 依赖安装

确保您的环境已安装 Python 3.8+，并安装以下依赖库：

```bash
pip install pygame numpy

```

### 运行仿真

在项目根目录下运行：

```bash
python main.py

```

### 操作指南 (Controls)

- **[空格键]**：在鼠标位置随机引燃火点 (模拟人为/突发火情)。
- **[鼠标左键]**：点击单元格可手动切换地形状态 (如建立阻火墙)。
- **UI 侧边栏**：
- 实时显示当前代数 (Gen) 与帧数 (Frame)。
- **ML Weights**：实时滚动的机器学习特征权重。
- **Logs**：任务分发日志与救援事件日志。

---

## 📊 算法细节 (Algorithm Details)

### 1. 机器学习特征工程 (6D Features)

UGV 在竞价时会评估以下特征向量 ：

1. ** Proximity**：距离接近度 (越近越好)
2. ** Severity**：火势严重度 (优先处理重灾区)
3. ** Battery**：剩余电量
4. ** Water**：剩余水量
5. ** Obstacle Density**：路径障碍密度
6. ** Wind Alignment**：风向对齐度 (上风口优势)

### 2. 遗传算法适应度函数 (Fitness Function)

为了防止算法早熟收敛，评估函数综合了以下 5 个维度：
$$Fitness = (N_{extinguish} \times 15) + (S_{bonus} \times 5) - (N_{stranded} \times 150) - (T_{crowded} \times 15) - (T_{idle} \times 8)$$
该公式强力惩罚机器人搁浅行为，并鼓励在减少拥挤的前提下最大化灭火产出。

---

## 📝 开发日志 (Dev Logs)

- **2025-12-26**: 重构 A\* 算法，移除显式 Closed Set，采用 Lazy Deletion 提升寻路效率。
- **2025-12-25**: 完善 SupportBot 逻辑，增加双倍速移动与路径自保清除机制。
- **2025-12-24**: 修正 Drone 扫描逻辑，校准视野范围。
- **2025-12-23**: 引入 GridMap 分层设计，分离燃料层 (Fuel) 与干燥度层 (Dryness)，加入自燃机制。

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
