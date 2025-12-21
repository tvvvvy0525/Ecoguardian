import pygame
import random
from agents.base_agent import BaseAgent
from configs.settings import COLOR_UAV, CELL_SIZE


class Drone(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_UAV)
        self.type = "UAV"
        self.scan_radius = 4
        self.target = None  # 巡逻目标

    def select_new_target(self, grid_map, frame_id):
        """蒙特卡洛探索：飞向最久没看（紧迫度最高）的区域"""
        best_score = -1
        best_target = None

        # 随机采样 20 个点评估覆盖率
        for _ in range(20):
            rx = random.randint(0, grid_map.width - 1)
            ry = random.randint(0, grid_map.height - 1)
            score = grid_map.get_average_urgency(rx, ry, self.scan_radius, frame_id)
            if score > best_score:
                best_score = score
                best_target = (rx, ry)

        self.target = best_target

    def step(self, grid_map, frame_id):
        """无人机飞行逻辑（无视地形）"""
        if not self.target:
            self.select_new_target(grid_map, frame_id)

        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y

        # 符号函数移动
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        self.move(step_x, step_y, grid_map.width, grid_map.height)

        # 到达附近则重选目标
        if abs(self.x - tx) + abs(self.y - ty) <= 1:
            self.select_new_target(grid_map, frame_id)

    def scan(self, grid_map, frame_id):
        """执行感知任务，返回发现的火点"""
        # 1. 告诉地图：这块地方已经巡视过，重置热力图
        grid_map.mark_scanned(self.x, self.y, self.scan_radius, frame_id)

        # 2. 感知范围内的火点
        found_fires = []
        for dx in range(-self.scan_radius, self.scan_radius + 1):
            for dy in range(-self.scan_radius, self.scan_radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if grid_map.get_state(nx, ny) == 2:
                    found_fires.append((nx, ny))
        return found_fires

    def draw(self, surface):
        px, py = self.x * CELL_SIZE, self.y * CELL_SIZE
        # 绘制无人机：菱形或圆形
        pygame.draw.circle(
            surface,
            self.color,
            (px + CELL_SIZE // 2, py + CELL_SIZE // 2),
            CELL_SIZE // 2 - 2,
        )
        # 绘制扫描范围阴影
        scan_rect = pygame.Rect(
            (self.x - self.scan_radius) * CELL_SIZE,
            (self.y - self.scan_radius) * CELL_SIZE,
            (self.scan_radius * 2 + 1) * CELL_SIZE,
            (self.scan_radius * 2 + 1) * CELL_SIZE,
        )
        s = pygame.Surface((scan_rect.width, scan_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            s, (0, 191, 255, 30), (0, 0, scan_rect.width, scan_rect.height)
        )
        surface.blit(s, (scan_rect.x, scan_rect.y))
