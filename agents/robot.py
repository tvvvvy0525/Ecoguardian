import pygame
from agents.base_agent import BaseAgent
from configs.settings import *
from core.pathfinding import astar


class Robot(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_UGV)
        self.status, self.target, self.current_path = "IDLE", None, []
        self.battery, self.water = ROBOT_MAX_BATTERY, ROBOT_MAX_WATER
        self.last_task_features = None

    def aoe_extinguish(self, grid_map):
        if self.water <= 0:
            return False
        ext = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                if grid_map.get_state(nx, ny) == 2:
                    grid_map.set_state(nx, ny, 6)
                    self.water -= 1
                    ext = True
                    if self.water <= 0:
                        return True
        return ext

    def step(self, grid_map, predictor=None):
        if self.battery <= 0:
            self.status = "STRANDED"
            return

        # [核心修复] 目标验证与惩罚逻辑
        if self.status == "MOVING" and self.target:
            target_state = grid_map.get_state(*self.target)
            if target_state != 2:  # 目标点不再是火（可能被抢或自然烧尽）
                if predictor and self.last_task_features:
                    # 给予负反馈 (Label 0)，让机器人学会这次决策是失败的
                    predictor.train(self.last_task_features, 0)
                # 停止脚步，重置状态
                self.status, self.target, self.current_path = "IDLE", None, []
                return

        # 全时 AOE
        if self.water > 0:
            if (
                self.aoe_extinguish(grid_map)
                and self.status == "MOVING"
                and predictor
                and self.last_task_features
            ):
                predictor.train(self.last_task_features, 1)  # 成功灭火的正反馈

        # 武装返航协议
        if self.status not in ["RETURNING", "STRANDED"] and (
            self.battery < ROBOT_LOW_BATTERY_THRESHOLD
            or self.water <= ROBOT_WATER_RESERVE
        ):
            depot = min(
                grid_map.depots, key=lambda d: abs(self.x - d[0]) + abs(self.y - d[1])
            )
            self.status, self.target = "RETURNING", None
            self.set_target(depot[0], depot[1], grid_map)

        # 移动逻辑
        if self.status in ["MOVING", "RETURNING"] and self.current_path:
            self.x, self.y = self.current_path.pop(0)
            self.battery -= 1
            if (self.x, self.y) == self.target:
                if self.status == "RETURNING":
                    self.battery, self.water = ROBOT_MAX_BATTERY, ROBOT_MAX_WATER
                self.status, self.target = "IDLE", None
        elif not self.current_path:
            self.status = "IDLE"

    def set_target(self, tx, ty, grid_map, feats=None):
        p = astar(grid_map, (self.x, self.y), (tx, ty), has_water=(self.water > 0))
        if p:
            self.target, self.current_path, self.last_task_features = (
                (tx, ty),
                p[1:],
                feats,
            )
            if self.status != "RETURNING":
                self.status = "MOVING"
            return True
        return False

    def calculate_bid(self, fire_pos, feats, predictor):
        dist = abs(self.x - fire_pos[0]) + abs(self.y - fire_pos[1])
        risk = (1.0 - predictor.predict_prob(feats)) * PREDICTION_PENALTY
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

        if self.path:
            self.x, self.y = self.path.pop(0)
            # 救援者自带 AOE 保护
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
