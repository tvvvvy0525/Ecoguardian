# agents/robot.py
from agents.base_agent import BaseAgent
from configs.settings import *
from core.pathfinding import astar
import math
import pygame
import random

class Robot(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_UGV)
        self.type = "UGV"
        self.target = None      
        self.current_path = []  
        self.status = "IDLE"    
        
        self.battery = ROBOT_MAX_BATTERY
        self.water = ROBOT_MAX_WATER
        self.idle_counter = 0
        self.replan_cooldown = 0
        
        self.last_task_features = None 

    def find_nearest_depot(self, grid_map):
        min_dist = float('inf')
        candidates = []
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                if grid_map.get_state(x, y) == 5:
                    dist = abs(self.x - x) + abs(self.y - y)
                    if dist < min_dist:
                        min_dist = dist
                        candidates = [(x, y)]
                    elif dist == min_dist:
                        candidates.append((x, y))
        if candidates:
            return random.choice(candidates)
        return None

    def scan_local(self, grid_map):
        local_fires = []
        scan_range = 2
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height:
                    if grid_map.get_state(nx, ny) == 2:
                        local_fires.append((nx, ny))
        return local_fires

    def set_target(self, tx, ty, grid_map, features=None):
        path = astar(grid_map, (self.x, self.y), (tx, ty))
        if path:
            self.target = (tx, ty)
            if features:
                self.last_task_features = features
            self.current_path = path[1:]
            if self.status == "IDLE":
                self.status = "MOVING"
            # print(f"Robot {self.id}: Path found to ({tx}, {ty}), len: {len(self.current_path)}")
            return True 
        else:
            # print(f"Robot {self.id}: No path to ({tx}, {ty}) - Unreachable")
            self.status = "IDLE"
            self.target = None
            return False 

    def find_immediate_fire(self, grid_map):
        scan_range = 3
        best_fire = None
        min_dist = float('inf')
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height:
                    if grid_map.get_state(nx, ny) == 2:
                        dist = abs(self.x - nx) + abs(self.y - ny)
                        if dist > 0 and dist < min_dist:
                            min_dist = dist
                            best_fire = (nx, ny)
        return best_fire

    def step(self, grid_map, predictor=None):
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1

        if self.status == "IDLE":
             if grid_map.get_state(self.x, self.y) != 5:
                self.idle_counter += 1
                if self.idle_counter > ROBOT_IDLE_RETURN_THRESHOLD:
                    depot = self.find_nearest_depot(grid_map)
                    if depot:
                        self.status = "RETURNING"
                        self.set_target(depot[0], depot[1], grid_map)
                        self.idle_counter = 0
             else:
                self.idle_counter = 0
        else:
            self.idle_counter = 0

        if self.status != "RETURNING":
            needs_refill = (self.battery < ROBOT_LOW_BATTERY_THRESHOLD) or (self.water < ROBOT_LOW_WATER_THRESHOLD)
            if needs_refill:
                depot = self.find_nearest_depot(grid_map)
                if depot:
                    self.status = "RETURNING"
                    self.set_target(depot[0], depot[1], grid_map)

        if self.status in ["MOVING", "RETURNING"]:
            if self.status == "MOVING" and self.target:
                tx, ty = self.target
                target_state = grid_map.get_state(tx, ty)
                
                # [关键修复: 动态补救机制]
                if target_state != 2:
                    nearby = self.find_immediate_fire(grid_map)
                    
                    if nearby:
                        # 找到了替补！原地改换目标
                        self.target = nearby 
                        self.current_path = [] 
                        return 
                    else:
                        # [关键修复: 选择性归因]
                        if target_state == 6:
                            # 被抢了 -> 负反馈
                            if self.last_task_features and predictor:
                                predictor.train(self.last_task_features, 0) 
                                self.last_task_features = None
                        elif target_state == 4:
                            # 自然烧完 -> 忽略
                            self.last_task_features = None 

                        self.status = "IDLE"
                        self.target = None
                        self.current_path = []
                        return 

            if len(self.current_path) > 0:
                next_pos = self.current_path[0]
                state = grid_map.get_state(next_pos[0], next_pos[1])
                is_blocked = (state == 3) or (state == 2 and next_pos != self.target)
                
                if is_blocked:
                    if self.replan_cooldown == 0:
                        new_path = astar(grid_map, (self.x, self.y), self.target)
                        if new_path:
                            self.current_path = new_path[1:]
                        else:
                            self.status = "IDLE"
                        self.replan_cooldown = 10
                    else:
                        return

                if len(self.current_path) > 0:
                    self.x, self.y = self.current_path.pop(0)
                    self.battery -= 1
            else:
                if grid_map.get_state(self.x, self.y) == 5:
                    self.battery = ROBOT_MAX_BATTERY
                    self.water = ROBOT_MAX_WATER
                    self.status = "IDLE"
                    self.target = None
                elif self.status == "MOVING": 
                    if self.target and (self.x, self.y) == self.target:
                        self.status = "EXTINGUISHING"
                    else:
                        self.status = "IDLE"
                        self.target = None
                else:
                    self.status = "IDLE"

        elif self.status == "EXTINGUISHING":
            if self.target:
                tx, ty = self.target
                if self.water > 0:
                    if grid_map.get_state(tx, ty) == 2:
                        grid_map.set_state(tx, ty, 6)
                        self.water -= 1 
                        
                        # 成功反馈 (Label 1)
                        if self.last_task_features and predictor:
                            predictor.train(self.last_task_features, 1) 
                            self.last_task_features = None
                            
                        nearby_fire = self.find_immediate_fire(grid_map)
                        if nearby_fire and self.water > 0:
                            self.set_target(nearby_fire[0], nearby_fire[1], grid_map)
                            return
                self.status = "IDLE"
                self.target = None

    def draw(self, surface):
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        if self.target and self.status in ["MOVING", "EXTINGUISHING"]:
            tx, ty = self.target
            start_pos = (px + CELL_SIZE//2, py + CELL_SIZE//2)
            end_pos = (tx * CELL_SIZE + CELL_SIZE//2, ty * CELL_SIZE + CELL_SIZE//2)
            pygame.draw.line(surface, self.color, start_pos, end_pos, width=2)
        rect = pygame.Rect(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(surface, self.color, rect)
        if self.status == "EXTINGUISHING":
            pygame.draw.rect(surface, (255, 140, 0), pygame.Rect(px, py, CELL_SIZE, CELL_SIZE), width=3)
        
        # 状态条绘制
        bar_bg_rect = pygame.Rect(px + 1, py - 4, STATUS_BAR_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(surface, (50, 0, 0), bar_bg_rect)
        bat_pct = max(0, self.battery / ROBOT_MAX_BATTERY)
        bat_width = int(STATUS_BAR_WIDTH * bat_pct)
        bat_color = (0, 255, 0) if bat_pct > 0.5 else ((255, 255, 0) if bat_pct > 0.2 else (255, 0, 0))
        pygame.draw.rect(surface, bat_color, pygame.Rect(px + 1, py - 4, bat_width, STATUS_BAR_HEIGHT))

        water_bg_rect = pygame.Rect(px + 1, py + CELL_SIZE + 1, STATUS_BAR_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(surface, (0, 0, 50), water_bg_rect)
        wat_pct = max(0, self.water / ROBOT_MAX_WATER)
        wat_width = int(STATUS_BAR_WIDTH * wat_pct)
        pygame.draw.rect(surface, (0, 191, 255), pygame.Rect(px + 1, py + CELL_SIZE + 1, wat_width, STATUS_BAR_HEIGHT))

    def calculate_bid(self, fire_pos, features, predictor):
        fx, fy = fire_pos
        dist = abs(self.x - fx) + abs(self.y - fy)
        battery_factor = (1.0 - (self.battery / 200.0)) * 50 
        base_cost = dist + battery_factor
        prob_success = predictor.predict_prob(features[0], features[1])
        risk_cost = (1.0 - prob_success) * PREDICTION_PENALTY
        return base_cost + risk_cost