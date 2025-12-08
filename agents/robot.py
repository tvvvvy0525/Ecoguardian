from agents.base_agent import BaseAgent
from configs.settings import COLOR_UGV, ROBOT_MAX_BATTERY, ROBOT_MAX_WATER, ROBOT_LOW_BATTERY_THRESHOLD, ROBOT_LOW_WATER_THRESHOLD, ROBOT_IDLE_RETURN_THRESHOLD
from core.pathfinding import astar
import math

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
        """寻找最近的补给站坐标"""
        min_dist = float('inf')
        nearest = None
        
        # 遍历地图找 State 5 (Depot)
        # 优化：因为补给站已知在四角，可以直接硬编码检查列表，但全图扫描更通用
        # 这里为了演示通用性，使用遍历 (实际应用可缓存 depot 列表)
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                if grid_map.get_state(x, y) == 5:
                    dist = abs(self.x - x) + abs(self.y - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)
        return nearest

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