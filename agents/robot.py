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
        self.replan_cooldown = 0

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

    def find_immediate_fire(self, grid_map):
        """寻找紧邻的（距离很近）的火点，用于连续作战"""
        scan_range = 3 # 搜索半径
        best_fire = None
        min_dist = float('inf')

        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                nx, ny = self.x + dx, self.y + dy
                
                # 越界检查
                if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height:
                    # 如果发现火点 (2)
                    if grid_map.get_state(nx, ny) == 2:
                        # 简单的距离计算
                        dist = abs(self.x - nx) + abs(self.y - ny)
                        # 排除掉自己刚刚灭掉的那个位置
                        if dist > 0 and dist < min_dist:
                            min_dist = dist
                            best_fire = (nx, ny)
        
        return best_fire

    def step(self, grid_map):
        """执行一步行动 (状态机核心)"""
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1

        # --- IDLE 计时逻辑 ---
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
            if self.status == "MOVING" and self.target:
                tx, ty = self.target
                target_state = grid_map.get_state(tx, ty)
                # 如果目标既不是火(2)，也不是我们要去的那个坐标(防止误判)
                if target_state != 2:
                    print(f"Robot {self.id}: Target fire at {self.target} is gone! Aborting.")
                    self.status = "IDLE"
                    self.target = None
                    self.current_path = []
                    return # 本帧直接结束

            if len(self.current_path) > 0:
                next_pos = self.current_path[0]
                
                # A. 检查路径是否被阻挡 (动态障碍检测)
                state = grid_map.get_state(next_pos[0], next_pos[1])
                # 阻挡条件: 墙(3)。注意：如果是去灭火，允许终点是火(2)；如果是回补给，允许终点是补给(5)
                # 简单起见，只要下一格是 3 就视为堵塞。如果是火但不是目标，也可以视为堵塞。
                is_blocked = (state == 3) or (state == 2 and next_pos != self.target)
                
                if is_blocked:
                    if self.replan_cooldown == 0:
                        print(f"Robot {self.id}: Path blocked! Re-planning...")
                        # 【修复1】这里修正了 astar 的调用，传入了 grid_map, start, end
                        new_path = astar(grid_map, (self.x, self.y), self.target)
                        if new_path:
                            self.current_path = new_path[1:]
                        else:
                            self.status = "IDLE" # 彻底放弃，防止卡死
                        self.replan_cooldown = 10 # 冷却 10 帧后再试
                    else:
                        # 冷却中，本帧原地等待
                        return

                # B. 执行移动
                # 再次确认路径非空 (因为重规划可能失败)
                if len(self.current_path) > 0:
                    self.x, self.y = self.current_path.pop(0)
                    self.battery -= 1 # 移动消耗电量
            
            else:
                # C. 路径走完了
                # 情况1: 到达补给站
                if grid_map.get_state(self.x, self.y) == 5:
                    print(f"Robot {self.id}: Refilled at Depot.")
                    self.battery = ROBOT_MAX_BATTERY
                    self.water = ROBOT_MAX_WATER
                    self.status = "IDLE"
                    self.target = None
                
                # 情况2: 准备灭火
                elif self.status == "MOVING": 
                    # 【修复2】双重检查：防止“隔空灭火”BUG
                    # 必须确认当前坐标真的等于目标坐标
                    if self.target and (self.x, self.y) == self.target:
                        self.status = "EXTINGUISHING"
                    else:
                        print(f"Robot {self.id}: Path ended but not at target! Aborting.")
                        self.status = "IDLE"
                        self.target = None
                
                # 其他情况
                else:
                    self.status = "IDLE"

        elif self.status == "EXTINGUISHING":
            if self.target:
                tx, ty = self.target
                if self.water > 0:
                    # 检查目标点还是不是火
                    if grid_map.get_state(tx, ty) == 2:
                        grid_map.set_state(tx, ty, 6) # 设为已扑灭
                        self.water -= 1 
                        print(f"Robot {self.id} extinguished fire at {tx}, {ty}")

                        # ==========================================
                        # 【修复3】 连续作战逻辑 (Chain Reaction)
                        # ==========================================
                        nearby_fire = self.find_immediate_fire(grid_map)
                        if nearby_fire and self.water > 0:
                            print(f"Robot {self.id}: Chain reaction! Moving to neighbor {nearby_fire}")
                            # 直接更新目标，不用经过调度中心
                            self.set_target(nearby_fire[0], nearby_fire[1], grid_map)
                            return # 直接进入下一帧处理，不要把 status 设为 IDLE
                        # ==========================================

                else:
                    print(f"Robot {self.id}: Out of water!")
                
                # 如果没水了，或者周围没火了，才停机
                self.status = "IDLE"
                self.target = None

    def draw(self, surface):
        """绘制机器人及其显眼的状态条"""
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        if self.target and self.status in ["MOVING", "EXTINGUISHING"]:
            tx, ty = self.target
            # 画一条从机器人中心到目标中心的线
            start_pos = (px + CELL_SIZE//2, py + CELL_SIZE//2)
            end_pos = (tx * CELL_SIZE + CELL_SIZE//2, ty * CELL_SIZE + CELL_SIZE//2)
            # 使用对应的颜色 (例如机器人颜色)
            pygame.draw.line(surface, self.color, start_pos, end_pos, width=2)
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

    def calculate_bid(self, fire_pos):
        """
        计算执行某任务的代价值 (Cost Function)
        Cost = 距离 + 电池惩罚 + 水量惩罚
        """
        fx, fy = fire_pos
        dist = abs(self.x - fx) + abs(self.y - fy)
        
        # 电池惩罚因子：电量越少，去远处的意愿越低
        # 如果电量是 100%，因子是 0；如果电量是 20%，因子很大
        battery_factor = (1.0 - (self.battery / 200.0)) * 50 
        
        return dist + battery_factor