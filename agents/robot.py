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
        self.status = "IDLE"    # IDLE, MOVING, EXTINGUISHING, RETURNING
        
        # --- 资源属性 ---
        self.battery = ROBOT_MAX_BATTERY
        self.water = ROBOT_MAX_WATER
        self.idle_counter = 0
    def find_nearest_depot(self, grid_map):
        """寻找最近的补给站坐标 (分散式策略)"""
        min_dist = float('inf')
        candidates = [] # 候选列表
        
        # 遍历地图找 State 5 (Depot)
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                if grid_map.get_state(x, y) == 5:
                    dist = abs(self.x - x) + abs(self.y - y)
                    
                    if dist < min_dist:
                        min_dist = dist
                        candidates = [(x, y)] # 发现更近的，重置列表
                    elif dist == min_dist:
                        candidates.append((x, y)) # 距离相等，加入列表
        
        if candidates:
            # 随机选择一个距离最近的，打破扎堆
            return random.choice(candidates)
        return None
    def scan_local(self, grid_map):
        """局部感知：探测周围 5x5 区域内的火点"""
        local_fires = []
        scan_range = 2 # 半径2，覆盖 5x5
        
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                nx, ny = self.x + dx, self.y + dy
                # 检查边界
                if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height:
                    if grid_map.get_state(nx, ny) == 2: # Fire
                        local_fires.append((nx, ny))
        return local_fires
    def set_target(self, tx, ty, grid_map):
        """设定目标并规划路径"""
        self.target = (tx, ty)
        path = astar(grid_map, (self.x, self.y), (tx, ty))
        
        if path:
            self.current_path = path[1:] # 去掉起点
            # 如果本来是 IDLE，设为 MOVING；如果是 RETURNING，保持 RETURNING
            if self.status == "IDLE":
                self.status = "MOVING"
            print(f"Robot {self.id}: Path found to ({tx}, {ty}), len: {len(self.current_path)}")
        else:
            print(f"Robot {self.id}: No path to ({tx}, {ty})")
            # 如果去不了补给站，就只能 IDLE 等死或等待障碍清除
            self.status = "IDLE"
            self.target = None

    def step(self, grid_map):
        """执行一步行动 (状态机核心)"""
        # --- 0. 局部感知与任务抢占 (新增核心逻辑) ---
        # 只要不是在回城(RETURNING)或已经没水了，就可以抢占任务
        if self.status != "RETURNING" and self.water > 0:
            local_fires = self.scan_local(grid_map)
            
            if local_fires:
                # 找最近的一个本地火点
                closest_fire = None
                min_dist = float('inf')
                for fx, fy in local_fires:
                    d = abs(self.x - fx) + abs(self.y - fy)
                    if d < min_dist:
                        min_dist = d
                        closest_fire = (fx, fy)
                
                # 决策：如果当前没有目标，或者当前目标太远(>5步)，而脚边就有火 -> 抢占！
                should_preempt = False
                if self.status == "IDLE":
                    should_preempt = True
                elif self.status == "MOVING" and self.target:
                    # 计算到当前目标的距离
                    current_target_dist = abs(self.x - self.target[0]) + abs(self.y - self.target[1])
                    # 如果本地火点比当前目标近得多，且不是同一个点
                    if min_dist < current_target_dist and closest_fire != self.target:
                        should_preempt = True
                
                if should_preempt and closest_fire:
                    print(f"Robot {self.id}: Preemptively targeting local fire at {closest_fire}")
                    # 直接切换目标，不经过调度中心
                    self.set_target(closest_fire[0], closest_fire[1], grid_map)
        if self.status == "IDLE":
            # 如果不在补给站(5)上，才需要计时
            if grid_map.get_state(self.x, self.y) != 5:
                self.idle_counter += 1
                if self.idle_counter > ROBOT_IDLE_RETURN_THRESHOLD:
                    depot = self.find_nearest_depot(grid_map)
                    if depot:
                        print(f"Robot {self.id}: Idle timeout ({self.idle_counter} frames). Returning to Depot.")
                        # 先切换状态，避免被 set_target 误判为普通移动
                        self.status = "RETURNING"
                        self.set_target(depot[0], depot[1], grid_map)
                        self.idle_counter = 0
            else:
                self.idle_counter = 0 # 已在补给站，不计时
        else:
            self.idle_counter = 0 # 工作中，重置计时
        # --- 1. 优先检查生存条件 (自动触发返航) ---
        # 只有在不是已经在回家的时候才检查
        if self.status != "RETURNING":
            needs_refill = (self.battery < ROBOT_LOW_BATTERY_THRESHOLD) or (self.water < ROBOT_LOW_WATER_THRESHOLD)
            if needs_refill:
                depot = self.find_nearest_depot(grid_map)
                if depot:
                    print(f"Robot {self.id}: Low resources (Bat:{self.battery}, Wat:{self.water}). Returning to Depot {depot}.")
                    self.status = "RETURNING"
                    self.set_target(depot[0], depot[1], grid_map)
                else:
                    print(f"Robot {self.id}: PANIC! No depot found!")

        # --- 2. 状态机执行 ---
        
        if self.status in ["MOVING", "RETURNING"]:
            if len(self.current_path) > 0:
                next_pos = self.current_path[0]
                
                # A. 检查路径是否被阻挡 (动态障碍检测)
                state = grid_map.get_state(next_pos[0], next_pos[1])
                # 阻挡条件: 墙(3)。注意：如果是去灭火，允许终点是火(2)；如果是回补给，允许终点是补给(5)
                # 简单起见，只要下一格是 3 就视为堵塞。如果是火但不是目标，也可以视为堵塞。
                is_blocked = (state == 3) or (state == 2 and next_pos != self.target)
                
                if is_blocked:
                    print(f"Robot {self.id}: Path blocked! Re-planning...")
                    # --- 动态重规划 (Re-planning) ---
                    # 尝试重新计算从当前位置到目标的路径
                    new_path = astar(grid_map, (self.x, self.y), self.target)
                    if new_path:
                        self.current_path = new_path[1:]
                        # 如果新路径的第一步还是堵的(极端情况)，这帧先不动
                        if not self.current_path: return 
                    else:
                        print(f"Robot {self.id}: Re-planning failed. Abort task.")
                        self.status = "IDLE"
                        self.target = None
                        self.current_path = []
                        return

                # B. 执行移动
                # 再次确认路径非空 (因为重规划可能失败)
                if len(self.current_path) > 0:
                    self.x, self.y = self.current_path.pop(0)
                    self.battery -= 1 # 移动消耗电量
            
            else:
                # C. 到达目的地
                # 情况1: 到达补给站
                if grid_map.get_state(self.x, self.y) == 5:
                    print(f"Robot {self.id}: Refilled at Depot.")
                    self.battery = ROBOT_MAX_BATTERY
                    self.water = ROBOT_MAX_WATER
                    self.status = "IDLE"
                    self.target = None
                
                # 情况2: 到达火点 (准备灭火)
                elif self.status == "MOVING": 
                    self.status = "EXTINGUISHING"
                
                # 其他情况
                else:
                    self.status = "IDLE"

        elif self.status == "EXTINGUISHING":
            if self.target:
                tx, ty = self.target
                # 检查是否还有水
                if self.water > 0:
                    if grid_map.get_state(tx, ty) == 2:
                        grid_map.set_state(tx, ty, 4) # 灭火 -> 焦土
                        self.water -= 1 # 消耗水
                        print(f"Robot {self.id} extinguished fire. Water left: {self.water}")
                    else:
                        # 火已经没了
                        pass
                else:
                    print(f"Robot {self.id}: Out of water!")
                    # 下一帧会被上面的资源检查捕获并返航
                
                # 任务结束
                self.status = "IDLE"
                self.target = None

    def draw(self, surface):
        """绘制机器人及其显眼的状态条"""
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        
        # 1. 绘制本体
        rect = pygame.Rect(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(surface, self.color, rect)
        
        # 2. 状态指示: 灭火中 (加粗橙色边框)
        if self.status == "EXTINGUISHING":
            pygame.draw.rect(surface, (255, 140, 0), pygame.Rect(px, py, CELL_SIZE, CELL_SIZE), width=3)

        # --- 3. 电量条 (Battery - 上方) ---
        # 背景条 (深红)
        bar_bg_rect = pygame.Rect(px + 1, py - 4, STATUS_BAR_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(surface, (50, 0, 0), bar_bg_rect)
        
        # 前景条 (根据电量变色: 绿 -> 黄 -> 红)
        bat_pct = max(0, self.battery / ROBOT_MAX_BATTERY)
        bat_width = int(STATUS_BAR_WIDTH * bat_pct)
        
        if bat_pct > 0.5: bat_color = (0, 255, 0)
        elif bat_pct > 0.2: bat_color = (255, 255, 0)
        else: bat_color = (255, 0, 0)
        
        pygame.draw.rect(surface, bat_color, pygame.Rect(px + 1, py - 4, bat_width, STATUS_BAR_HEIGHT))

        # --- 4. 水量条 (Water - 下方) ---
        # 背景条 (深蓝)
        water_bg_rect = pygame.Rect(px + 1, py + CELL_SIZE + 1, STATUS_BAR_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(surface, (0, 0, 50), water_bg_rect)
        
        # 前景条 (青色)
        wat_pct = max(0, self.water / ROBOT_MAX_WATER)
        wat_width = int(STATUS_BAR_WIDTH * wat_pct)
        pygame.draw.rect(surface, (0, 191, 255), pygame.Rect(px + 1, py + CELL_SIZE + 1, wat_width, STATUS_BAR_HEIGHT))