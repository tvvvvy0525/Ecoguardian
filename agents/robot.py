import pygame
import numpy as np
from agents.base_agent import BaseAgent
from configs.settings import *
from core.pathfinding import astar

class Robot(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_UGV)
        self.status = "IDLE" # 机器人状态
        self.target = None # 机器人目标
        self.current_path = [] # 机器人当前路径
        self.battery = ROBOT_MAX_BATTERY # 机器人电池电量
        self.water = ROBOT_MAX_WATER # 机器人水资源
        self.last_task_features = None # 机器人上次任务特征
        self.idle_timer = 0 # 机器人闲置时间

    def aoe_extinguish(self, grid_map, current_genome=None):
        """执行 3x3 区域灭火：遍历机器人周围3x3范围内的区域，如果区域内有火点，且机器人有水资源，则灭火"""
        if self.water <= 0:
            return False
        ext = False # 是否灭火成功
        for dx in [-1, 0, 1]: # 遍历机器人周围3x3范围内的区域
            for dy in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                if grid_map.get_state(nx, ny) == 2:
                    if current_genome:
                        x_min, x_max = max(0, nx - 1), min(grid_map.width, nx + 2)
                        y_min, y_max = max(0, ny - 1), min(grid_map.height, ny + 2)
                        sev = np.sum(grid_map.grid[x_min:x_max, y_min:y_max] == 2)
                        current_genome.severity_bonus += sev # 增加火点严重度奖励
                    grid_map.set_state(nx, ny, 6)
                    self.water -= 1
                    ext = True
                    if self.water <= 0:
                        return True
        return ext

    def find_local_fire(self, grid_map, neighbors=None, search_radius=6, dynamic_radius=2):
        """
        [自主决策] 寻找附近的火点，同时严格遵守社交距离
        """
        candidates = []
        social_r = dynamic_radius  # 保持与 main.py 一致的避嫌半径
        # 遍历机器人周围search_radius范围内的区域，如果区域内有火点，则记录火点坐标
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = self.x + dx, self.y + dy

                if grid_map.get_state(nx, ny) == 2:

                    # --- 避嫌检查 ---
                    is_taken = False
                    if neighbors:
                        for other_bot in neighbors:
                            if other_bot.id == self.id:
                                continue

                            # 1. 检查队友的目标 (Target) - 防止撞车
                            if other_bot.target:
                                t_dist = abs(other_bot.target[0] - nx) + abs(
                                    other_bot.target[1] - ny
                                )
                                if t_dist <= social_r: # 如果队友目标与当前火点距离小于等于避嫌半径，则认为该火点被占用
                                    is_taken = True
                                    break

                            # 2. 检查队友的当前位置 (Position) - 防止扎堆
                            p_dist = abs(other_bot.x - nx) + abs(other_bot.y - ny)
                            if p_dist <= social_r: # 如果队友当前位置与当前火点距离小于等于避嫌半径，则认为该火点被占用
                                is_taken = True
                                break
                    # 如果该火点没有被占用，则记录火点坐标
                    if not is_taken:
                        dist = abs(dx) + abs(dy) # 计算当前火点与机器人距离
                        candidates.append((dist, (nx, ny)))
        # 如果找到的火点不为空，则返回距离最近的火点坐标
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    def step(self, grid_map, predictor=None, neighbors=None, current_genome=None):
        if self.battery <= 0:
            self.status = "STRANDED" # 机器人电量不足，进入 stranded 状态
            return
        # 1. 闲置超时返航：如果机器人闲置时间超过阈值，则返回最近的 depot
        if self.status == "IDLE":
            self.idle_timer += 1
            if self.idle_timer >= ROBOT_IDLE_RETURN_THRESHOLD: # 如果机器人闲置时间超过阈值
                depot = min(grid_map.depots, key=lambda d: abs(self.x - d[0]) + abs(self.y - d[1])) # 找到距离最近的 depot
                if (self.x, self.y) != depot: # 如果机器人当前位置不等于 depot，则返回 depot
                    self.status = "RETURNING" # 机器人进入 returning 状态
                    self.set_target(depot[0], depot[1], grid_map) # 设置机器人目标为 depot
                    self.idle_timer = 0 # 重置闲置时间
        # 2. 目标有效性验证：如果机器人目标状态为 moving，且目标不为空，则验证目标有效性
        if self.status == "MOVING" and self.target:
            target_state = grid_map.get_state(*self.target) # 获取目标状态
            if target_state != 2:
                if predictor and self.last_task_features: # 如果预测器和上次任务特征不为空，则训练预测器
                    predictor.train(self.last_task_features, 0) # 训练预测器
                self.status, self.target, self.current_path = "IDLE", None, [] # 机器人进入 idle 状态
                return
        # 3. 边走边灭：执行 3x3 区域灭火
        self.aoe_extinguish(grid_map, current_genome=current_genome)
        # 4. 武装返航协议
        if self.status not in ["RETURNING", "STRANDED"] and (
            self.battery < ROBOT_LOW_BATTERY_THRESHOLD
            or self.water <= ROBOT_WATER_RESERVE
        ):
            depot = min(
                grid_map.depots, key=lambda d: abs(self.x - d[0]) + abs(self.y - d[1])
            )
            self.status, self.target = "RETURNING", None
            self.set_target(depot[0], depot[1], grid_map)

        # 5. 移动逻辑
        if self.status in ["MOVING", "RETURNING"] and self.current_path:
            self.x, self.y = self.current_path.pop(0)
            self.battery -= 1

            # --- 到达逻辑 ---
            if (self.x, self.y) == self.target:
                if self.status == "RETURNING":
                    self.battery, self.water = ROBOT_MAX_BATTERY, ROBOT_MAX_WATER
                    self.status, self.target = "IDLE", None
                else:
                    if predictor and self.last_task_features:
                        predictor.train(self.last_task_features, 1)
                    # 到达火点后，尝试连击 (Mopping Up)
                    local_fire = None
                    if (
                        self.water > ROBOT_WATER_RESERVE
                        and self.battery > ROBOT_LOW_BATTERY_THRESHOLD
                    ):
                        # 传入 neighbors 进行避嫌
                        local_fire = self.find_local_fire(grid_map, neighbors, dynamic_radius=current_genome.radius)

                    if local_fire:
                        self.set_target(
                            local_fire[0],
                            local_fire[1],
                            grid_map,
                            self.last_task_features,
                        )
                    else:
                        self.status, self.target = "IDLE", None

        elif not self.current_path:
            self.status = "IDLE"

    def set_target(self, tx, ty, grid_map, feats=None):
        path = astar(grid_map, (self.x, self.y), (tx, ty), has_water=(self.water > 0))
        if path:
            self.target, self.current_path, self.last_task_features = (
                (tx, ty),
                path[1:],
                feats,
            )
            if self.status != "RETURNING":
                self.status = "MOVING"
            return True
        return False

    def calculate_bid(self, fire_pos, feats, predictor, penalty=PREDICTION_PENALTY):
        dist = abs(self.x - fire_pos[0]) + abs(self.y - fire_pos[1])
        prob = predictor.predict_prob(feats)
        risk = (1.0 - prob) * penalty
        return dist + risk + (1.0 - self.battery / ROBOT_MAX_BATTERY) * 50

    def draw(self, surface):
        px, py = self.x * CELL_SIZE, self.y * CELL_SIZE
        if self.current_path and self.target:
            pts = [(self.x * CELL_SIZE + 10, self.y * CELL_SIZE + 10)] + [
                (p[0] * CELL_SIZE + 10, p[1] * CELL_SIZE + 10)
                for p in self.current_path
            ]
            if len(pts) > 1:
                pygame.draw.lines(surface, self.color, False, pts, 1)

        rect_color = self.color if self.status != "STRANDED" else (100, 100, 100)
        pygame.draw.rect(
            surface, rect_color, (px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        )
        pygame.draw.rect(
            surface,
            (0, 255, 0),
            (px + 1, py - 3, int(18 * self.battery / ROBOT_MAX_BATTERY), 2),
        )
        pygame.draw.rect(
            surface,
            (0, 191, 255),
            (px + 1, py + CELL_SIZE + 1, int(18 * self.water / ROBOT_MAX_WATER), 2),
        )
        if self.status == "IDLE":
            font = pygame.font.SysFont("Arial", 10)
            text = font.render("Wait", True, (255, 255, 255))
            surface.blit(text, (px, py - 12))


class SupportBot(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_SUPPORT)
        self.target_robot = None
        self.path = []

    def step(self, grid_map, all_robots):
        if not self.target_robot:
            stranded = [r for r in all_robots if r.status == "STRANDED"]
            if stranded:
                self.target_robot = min(
                    stranded, key=lambda r: abs(self.x - r.x) + abs(self.y - r.y)
                )
                self.path = (
                    astar(
                        grid_map,
                        (self.x, self.y),
                        (self.target_robot.x, self.target_robot.y),
                        True,
                    )
                    or []
                )
        for _ in range(2):
            if self.path:
                self.x, self.y = self.path.pop(0)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if grid_map.get_state(self.x + dx, self.y + dy) == 2:
                            grid_map.set_state(self.x + dx, self.y + dy, 6)

                if (
                    abs(self.x - self.target_robot.x) <= 1
                    and abs(self.y - self.target_robot.y) <= 1
                ):
                    self.target_robot.battery = ROBOT_MAX_BATTERY
                    self.target_robot.water = ROBOT_MAX_WATER
                    self.target_robot.status = "IDLE"
                    self.target_robot = None
                    self.path = []
                    break
