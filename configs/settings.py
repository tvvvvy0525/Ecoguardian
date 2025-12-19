# configs/settings.py

# --- 屏幕与网格配置 ---
GRID_WIDTH = 40        # 网格宽度 (列数)
GRID_HEIGHT = 30       # 网格高度 (行数)
CELL_SIZE = 20         # 每个格子的像素大小
SIDEBAR_WIDTH = 200    # 右侧信息栏宽度

# 窗口总尺寸
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60           # 帧率 (仿真速度)

# --- 颜色定义 (RGB) ---
COLOR_BG = (30, 30, 30)         # 背景黑灰
COLOR_GRID_LINE = (50, 50, 50)  # 网格线颜色

# 单元格颜色映射
COLOR_EMPTY = (240, 240, 240)   # 空地: 白色
COLOR_TREE = (34, 139, 34)      # 树木: 森林绿
COLOR_FIRE = (255, 69, 0)       # 火焰: 红橙色
COLOR_BURNT = (50, 50, 50)      # 焦土: 深灰
COLOR_WALL = (100, 100, 100)    # 障碍: 灰色
COLOR_EXTINGUISHED = (100, 149, 237) # 熄灭的火: 蓝色

# 智能体颜色
COLOR_UAV = (0, 191, 255)       # 无人机: 深天蓝
COLOR_UGV = (255, 215, 0)       # 机器人: 金色
COLOR_DEPOT = (138, 43, 226)    # 补给站: 紫色

# --- 仿真参数 ---
FIRE_SPREAD_PROB = 0.01         # 火势向四周蔓延的概率
TREE_DENSITY = 0.7              # 初始森林覆盖率

# --- 物理引擎参数 ---
TREE_MAX_FUEL = 100              # 树木燃料值 (持续燃烧帧数)
WIND_STRENGTH = 1.5             # 风力影响因子 (建议 0.0 ~ 2.0)

ROBOT_MAX_BATTERY = 200         # 最大电量 (移动消耗)
ROBOT_MAX_WATER = 20            # 最大水量 (灭火消耗)
ROBOT_LOW_BATTERY_THRESHOLD = 50 # 低电量阈值 (触发返航)
ROBOT_LOW_WATER_THRESHOLD = 5    # 低水量阈值 (触发返航)
ROBOT_IDLE_RETURN_THRESHOLD = 100 # 空闲自动回补给站阈值 (帧数)

DRYNESS_INCREASE_RATE = 1.5     # 每帧增加的干燥度
IGNITION_DRYNESS_THRESHOLD = 100 # 超过此干燥度可能自燃
SPONTANEOUS_FIRE_PROB = 0.0001   # 超过阈值后的每帧自燃概率

STATUS_BAR_WIDTH = 18           # 状态条宽度
STATUS_BAR_HEIGHT = 3           # 状态条高度

# --- [新增] 机器学习预测参数 (修正版) ---
ML_LEARNING_RATE = 0.1          # 学习率
BID_REJECT_THRESHOLD = 800.0    # 拒绝接单的代价阈值 (软阈值)
PREDICTION_PENALTY = 500.0      # [Modified] 从 1000 降为 500，降低恐惧影响