from agents.base_agent import BaseAgent
from configs.settings import COLOR_UAV, GRID_WIDTH, GRID_HEIGHT
import random

class Drone(BaseAgent):
    def __init__(self, agent_id, x, y):
        super().__init__(agent_id, x, y, COLOR_UAV)
        self.type = "UAV"
        self.scan_radius = 4  
        self.target = None # 当前导航目标
    
    def select_new_target(self, grid_map, frame_id):
        """蒙特卡洛采样：随机找几个点，去那个'最久没看'的地方"""
        best_score = -1
        best_target = None
        
        # 采样 20 个随机点进行评估
        # 这种方法比遍历全图要快得多，且能保证随机性
        for _ in range(20):
            rx = random.randint(0, grid_map.width - 1)
            ry = random.randint(0, grid_map.height - 1)
            
            score = grid_map.get_average_urgency(rx, ry, self.scan_radius, frame_id)
            
            if score > best_score:
                best_score = score
                best_target = (rx, ry)
        
        self.target = best_target
        # print(f"Drone {self.id} heading to {self.target} (Score: {best_score:.1f})")

    def step(self, grid_map, frame_id):
        """智能飞行逻辑"""
        # 如果没有目标，选一个
        if not self.target:
            self.select_new_target(grid_map, frame_id)
            
        # 移动逻辑：向目标靠近
        tx, ty = self.target
        dx = tx - self.x
        dy = ty - self.y
        
        # 简单的网格移动：X轴和Y轴各走一步（如果需要）
        # sign函数逻辑
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        
        # 无人机可以飞越障碍，所以只需检查边界 (BaseAgent.move 已包含边界检查)
        self.move(step_x, step_y, grid_map.width, grid_map.height)
        
        # 判断是否到达（或非常接近）
        dist = abs(self.x - tx) + abs(self.y - ty)
        if dist <= 2: # 到达附近就换目标，保持流动性
            self.select_new_target(grid_map, frame_id)

    def scan(self, grid_map, frame_id):
        """感知：标记热力图并返回发现的火点"""
        # 1. 告诉地图：这块地方我看过了
        grid_map.mark_scanned(self.x, self.y, self.scan_radius, frame_id)
        
        # 2. 原有的探测逻辑
        found_fires = []
        for dx in range(-self.scan_radius, self.scan_radius + 1):
            for dy in range(-self.scan_radius, self.scan_radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if grid_map.get_state(nx, ny) == 2: # Fire
                    found_fires.append((nx, ny))
        return found_fires