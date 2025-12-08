import numpy as np
import random
from configs.settings import *

class GridMap:
    # 定义风向映射表 (静态类属性)
    WIND_DATA = [
        ("N", (0, -1)), ("S", (0, 1)), 
        ("W", (-1, 0)), ("E", (1, 0)),
        ("NW", (-1, -1)), ("NE", (1, -1)),
        ("SW", (-1, 1)), ("SE", (1, 1))
    ]
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        
        # 状态定义
        # 0: Empty, 1: Tree, 2: Fire, 3: Wall, 4: Burnt
        self.grid = np.zeros((width, height), dtype=int)
        # 燃料网格：存储每个格子的剩余燃料
        self.fuel_grid = np.zeros((width, height), dtype=int)

        # 记录每个格子最后一次被无人机覆盖的帧数 (初始化为0)
        self.last_scan_frame = np.zeros((width, height), dtype=int)

        self.wind_name, self.wind_direction = random.choice(self.WIND_DATA)
        print(f"Simulation Init: Wind is blowing {self.wind_name} {self.wind_direction}")
        
        self.generate_forest()

    def generate_forest(self, density=TREE_DENSITY):
        """随机生成森林和障碍物"""
        self.fuel_grid.fill(0) # 重置燃料
        for x in range(self.width):
            for y in range(self.height):
                rand = random.random()
                if rand < density:
                    self.grid[x][y] = 1  # Tree
                    self.fuel_grid[x][y] = TREE_MAX_FUEL # 初始化燃料
                elif rand < density + 0.05:
                    self.grid[x][y] = 3  # Wall (少量障碍物)
                else:
                    self.grid[x][y] = 0  # Empty
        depots = [
            (0, 0), 
            (self.width - 1, 0),
            (0, self.height - 1), 
            (self.width - 1, self.height - 1)
        ]
        for dx, dy in depots:
            # 清除补给站位置的障碍和树，确保安全
            self.grid[dx][dy] = 5
            self.fuel_grid[dx][dy] = 0 # 补给站不可燃

    def ignite_random(self):
        """随机点燃一棵树"""
        trees = np.argwhere(self.grid == 1)
        if len(trees) > 0:
            idx = random.randint(0, len(trees) - 1)
            x, y = trees[idx]
            self.grid[x][y] = 2  # Set to Fire
            print(f"Fire started at ({x}, {y})")
            return (x, y)
        return None
    
    def update_fire_spread(self):
        """核心逻辑：模拟火势蔓延 (带风向与燃料)"""
        # 复制当前网格，确保同步更新
        new_grid = self.grid.copy()
        
        # 获取所有当前着火点的坐标
        fire_indices = np.argwhere(self.grid == 2)
        
        for fx, fy in fire_indices:
            # 1. 燃料消耗与自动熄灭逻辑
            self.fuel_grid[fx][fy] -= 1
            if self.fuel_grid[fx][fy] <= 0:
                new_grid[fx][fy] = 4 # 燃料耗尽，转为焦土
                continue # 已熄灭，不再向周围传播
            
            # 2. 尝试引燃周围 8 邻域的树木
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    
                    nx, ny = fx + dx, fy + dy
                    # 边界检查
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        # 如果邻居是树
                        if self.grid[nx][ny] == 1:
                            # --- 风向逻辑开始 ---
                            # 计算传播方向与风向的点积
                            # spread vector = (dx, dy), wind vector = WIND_DIRECTION
                            dot_prod = dx * self.wind_direction[0] + dy * self.wind_direction[1]
                            # 动态调整概率: 基础概率 * (1 + 风力因子 * 点积)
                            # 顺风(dot>0)概率增加，逆风(dot<0)概率减少
                            factor = 1.0 + (dot_prod * WIND_STRENGTH)
                            current_prob = FIRE_SPREAD_PROB * factor
                            
                            if current_prob < 0: 
                                current_prob = 0
                            # --- 风向逻辑结束 ---

                            if random.random() < current_prob:
                                new_grid[nx][ny] = 2 # 点燃

        self.grid = new_grid

    def get_state(self, x, y):
        """获取指定坐标的状态"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return 3 # 越界视为墙

    def set_state(self, x, y, state):
        """设置状态"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y] = state

    def get_fire_locations(self):
        """返回所有火点的坐标列表"""
        return np.argwhere(self.grid == 2)
    
    def mark_scanned(self, cx, cy, radius, frame_id):
        """标记某个区域在当前帧已被扫描"""
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius + 1)
        
        # NumPy 切片批量更新
        self.last_scan_frame[x_min:x_max, y_min:y_max] = frame_id

    def get_average_urgency(self, cx, cy, radius, frame_id):
        """计算某个区域的平均'紧迫度' (越久没看，紧迫度越高)"""
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius + 1)
        
        # 提取该区域的历史帧数据
        area_history = self.last_scan_frame[x_min:x_max, y_min:y_max]
        
        # 紧迫度 = 当前帧 - 上次看到的帧 (差值越大越需要看)
        urgency_score = np.mean(frame_id - area_history)
        return urgency_score