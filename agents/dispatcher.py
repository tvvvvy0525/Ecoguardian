# agents/dispatcher.py
import numpy as np

class Dispatcher:
    def __init__(self):
        self.known_fires = set() # 全局已知的火点集合
        self.assigned_tasks = {} # 记录任务分配情况 {robot_id: fire_pos}

    def update_perception(self, drones):
        """1. 信息汇聚：收集所有无人机的发现"""
        for drone in drones:
            # 假设无人机有 found_fires 属性或是通过某种通信方式上报
            # 这里我们直接访问无人机上一帧扫描到的结果（需要修改Drone类稍作存储）
            # 或者在 main loop 中传入
            pass 
        # (注：实际逻辑在 main.py 中通过 detected_fires 集合传入更方便)

    def assign_tasks(self, robots, detected_fires, grid_map):
        """2. 协作决策：基于效用的任务分配"""
        
        # 筛选出有效的、未被分配的火点
        active_fires = [f for f in detected_fires if f not in self.assigned_tasks.values()]
        
        # 筛选出空闲的机器人
        idle_robots = [r for r in robots if r.status == "IDLE"]

        if not active_fires or not idle_robots:
            return

        # 简单的贪心策略升级为：为每个火点寻找最优机器人
        # 或者：为每个机器人寻找最优火点
        # 这里采用：遍历任务，寻找最佳匹配
        
        for fire in active_fires:
            best_robot = None
            min_cost = float('inf')

            for robot in idle_robots:
                # 只有电量和水量健康的机器人才参与竞标
                if robot.battery > 30 and robot.water > 0:
                    cost = robot.calculate_bid(fire)
                    if cost < min_cost:
                        min_cost = cost
                        best_robot = robot
            
            # 如果找到了合适的机器人，分配任务
            if best_robot:
                best_robot.set_target(fire[0], fire[1], grid_map)
                self.assigned_tasks[best_robot.id] = fire
                idle_robots.remove(best_robot) # 该机器人不再空闲
                
        # 清理已完成的任务
        self.cleanup_finished_tasks(robots)

    def cleanup_finished_tasks(self, robots):
        """清理状态"""
        # 如果机器人变回 IDLE，说明任务完成或取消，从记录中移除
        for r in robots:
            if r.status == "IDLE" and r.id in self.assigned_tasks:
                del self.assigned_tasks[r.id]