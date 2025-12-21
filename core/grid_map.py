import numpy as np
import random
from configs.settings import *


class GridMap:
    WIND_DATA = [
        ("N", (0, -1)),
        ("S", (0, 1)),
        ("W", (-1, 0)),
        ("E", (1, 0)),
        ("NW", (-1, -1)),
        ("NE", (1, -1)),
        ("SW", (-1, 1)),
        ("SE", (1, 1)),
    ]

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=int)
        self.fuel_grid = np.zeros((width, height), dtype=int)
        self.last_scan_frame = np.zeros((width, height), dtype=int)
        self.dryness_grid = np.zeros((width, height), dtype=float)
        self.wind_name, self.wind_direction = random.choice(self.WIND_DATA)
        self.depots = []  # [新增] 补给站索引
        print(
            f"Simulation Init: Wind is blowing {self.wind_name} {self.wind_direction}"
        )
        self.generate_forest()

    def generate_forest(self, density=TREE_DENSITY):
        self.fuel_grid.fill(0)
        for x in range(self.width):
            for y in range(self.height):
                rand = random.random()
                if rand < density:
                    self.grid[x][y] = 1
                    self.fuel_grid[x][y] = TREE_MAX_FUEL
                    self.dryness_grid[x][y] = random.uniform(
                        0, IGNITION_DRYNESS_THRESHOLD * 0.5
                    )
                elif rand < density + 0.05:
                    self.grid[x][y] = 3
                else:
                    self.grid[x][y] = 0
        depot_coords = [
            (0, 0),
            (self.width - 1, 0),
            (0, self.height - 1),
            (self.width - 1, self.height - 1),
        ]
        for dx, dy in depot_coords:
            self.grid[dx][dy] = 5
            self.fuel_grid[dx][dy] = 0
            self.depots.append((dx, dy))  # [清单2] 记录坐标

    # [清单3] 向量化更新
    def update_dryness(self):
        tree_mask = self.grid == 1
        noise = np.random.uniform(0.5, 1.5, size=(self.width, self.height))
        self.dryness_grid[tree_mask] += DRYNESS_INCREASE_RATE * noise[tree_mask]

        ignite_mask = tree_mask & (self.dryness_grid > IGNITION_DRYNESS_THRESHOLD)
        for x, y in np.argwhere(
            ignite_mask
            & (np.random.random((self.width, self.height)) < SPONTANEOUS_FIRE_PROB)
        ):
            self.grid[x][y] = 2
            self.dryness_grid[x][y] = 0
            print(f"Spontaneous ignition at ({x}, {y})!")

    # [清单3] 向量化核心
    def update_fire_spread(self):
        self.update_dryness()
        new_grid = self.grid.copy()
        fire_mask = self.grid == 2
        self.fuel_grid[fire_mask] -= 1
        new_grid[fire_mask & (self.fuel_grid <= 0)] = 4

        fire_indices = np.argwhere(self.grid == 2)
        for fx, fy in fire_indices:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = fx + dx, fy + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and self.grid[nx][ny] == 1
                    ):
                        dot_prod = (
                            dx * self.wind_direction[0] + dy * self.wind_direction[1]
                        )
                        factor = 1.0 + (dot_prod * WIND_STRENGTH)
                        if random.random() < FIRE_SPREAD_PROB * max(0, factor):
                            new_grid[nx][ny] = 2
                            self.dryness_grid[nx][ny] = 0
        self.grid = new_grid

    # --- 以下完全保持原始逻辑与格式 ---
    def ignite_random(self):
        """随机点燃一棵树"""
        trees = np.argwhere(self.grid == 1)
        if len(trees) > 0:
            idx = random.randint(0, len(trees) - 1)
            x, y = trees[idx]
            self.grid[x][y] = 2  # 点燃树木
            self.dryness_grid[x][y] = 0  # 初始干燥度
            print(f"Fire started at ({x}, {y})")
            return (x, y)
        return None

    def get_state(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return 3

    def set_state(self, x, y, state):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y] = state
            if state == 4 or state == 2:
                self.dryness_grid[x][y] = random.uniform(0, 5)

    def mark_scanned(self, cx, cy, radius, frame_id):
        x_min, x_max = max(0, cx - radius), min(self.width, cx + radius + 1)
        y_min, y_max = max(0, cy - radius), min(self.height, cy + radius + 1)
        self.last_scan_frame[x_min:x_max, y_min:y_max] = frame_id

    def get_average_urgency(self, cx, cy, radius, frame_id):
        x_min, x_max = max(0, cx - radius), min(self.width, cx + radius + 1)
        y_min, y_max = max(0, cy - radius), min(self.height, cy + radius + 1)
        return np.mean(frame_id - self.last_scan_frame[x_min:x_max, y_min:y_max])
