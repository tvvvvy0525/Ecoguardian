# main.py
import pygame
import sys
import numpy as np
from configs.settings import *
from core.grid_map import GridMap
from agents.drone import Drone
from agents.robot import Robot

# 初始化 Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("EcoGuardian - Forest Fire Simulation")
clock = pygame.time.Clock()

COLOR_MAP = {
    0: COLOR_EMPTY,
    1: COLOR_TREE,
    2: COLOR_FIRE,
    3: COLOR_WALL,
    4: COLOR_BURNT,
    5: COLOR_DEPOT
}

def draw_grid(surface, grid_map):
    for x in range(grid_map.width):
        for y in range(grid_map.height):
            state = grid_map.grid[x][y]
            color = COLOR_MAP.get(state, COLOR_EMPTY)
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)

def draw_sidebar(surface, grid_map, robots):
    """侧边栏显示更多信息"""
    sidebar_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, (40, 40, 40), sidebar_rect)
    font = pygame.font.SysFont("Arial", 16)
    
    # 统计
    trees = np.sum(grid_map.grid == 1)
    fires = np.sum(grid_map.grid == 2)
    burnt = np.sum(grid_map.grid == 4)
    
    # 计算空闲机器人
    idle_bots = sum(1 for r in robots if r.status == "IDLE")
    
    texts = [
        f"EcoGuardian System",
        f"----------------",
        f"Wind: {grid_map.wind_name}",  #显示风向
        f"Strength: {WIND_STRENGTH}x", #显示风力
        f"----------------",
        f"Trees: {trees}",
        f"Fires: {fires} (!)",
        f"Burnt: {burnt}",
        f"----------------",
        f"Robots Total: {len(robots)}",
        f"Robots Idle: {idle_bots}",
        f"----------------",
        f"KEYS:",
        f"[SPACE]: Ignite",
        f"[R]: Reset"
    ]
    
    for i, text in enumerate(texts):
        # 简单的警告色
        color = (255, 100, 100) if "(!)" in text and fires > 0 else (200, 200, 200)
        if "Wind:" in text: color = (100, 200, 255) 
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (GRID_WIDTH * CELL_SIZE + 10, 20 + i * 25))

def main():
    env_map = GridMap()
    env_map.ignite_random() # 开局一把火

    # --- 1. 初始化智能体 ---
    drones = [
        Drone(0, 0, 0),
        Drone(1, GRID_WIDTH-1, GRID_HEIGHT-1) # 初始分开一点
    ]
    
    robots = [
        Robot(101, 2, GRID_HEIGHT-2),
        Robot(102, 5, GRID_HEIGHT-2),
        Robot(103, GRID_WIDTH-2, GRID_HEIGHT-2)
    ]
    
    # 任务队列 (Set 防止重复)
    detected_fires = set()
    frame_count = 0 # 帧计数器

    running = True
    while running:
        frame_count += 1 # 每一帧累加
        # --- 2. 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env_map = GridMap()
                    detected_fires.clear()
                elif event.key == pygame.K_SPACE:
                    env_map.ignite_random()

        # --- 3. 智能体逻辑更新 ---
        
        if env_map: # 简单的非空检查
             # A. 环境演化
            env_map.update_fire_spread()
            
        # B. 无人机感知 (更新了调用参数)
        current_frame_fires = set()
        for drone in drones:
            # 无人机每帧移动一次即可 (原代码是两次，现在改为一次以匹配 10FPS 节奏)
            drone.step(env_map, frame_count) 
            
            # 扫描并更新热力图
            found = drone.scan(env_map, frame_count)
            for f in found:
                current_frame_fires.add(f)
        for robot in robots:
            local_found = robot.scan_local(env_map)
            for f in local_found:
                current_frame_fires.add(f)
        detected_fires.update(current_frame_fires)
        
        # 移除已经不再燃烧的火点 (可能已经被灭了)
        # 注意：这里需要做一个清洗，否则任务池会无限膨胀
        valid_fires = set()
        for fx, fy in detected_fires:
            if env_map.get_state(fx, fy) == 2:
                valid_fires.add((fx, fy))
        detected_fires = valid_fires

        # C. 机器人调度 (贪心算法)
        # 遍历所有机器人
        for robot in robots:
            # 只有 IDLE 状态才接受新灭火任务
            # 如果是 MOVING(去灭火), EXTINGUISHING(灭火中), RETURNING(回补给) 都不接受调度
            if robot.status == "IDLE" and detected_fires:
                
                # 再次检查电量/水量，防止刚 IDLE 但资源不足的情况 (双重保险)
                if robot.battery < ROBOT_LOW_BATTERY_THRESHOLD or robot.water <= 0:
                     # step() 会处理返航，这里先跳过分配
                     pass 
                else:
                    best_fire = None
                    min_dist = float('inf')
                    
                    for fx, fy in detected_fires:
                        dist = abs(robot.x - fx) + abs(robot.y - fy)
                        if dist < min_dist:
                            min_dist = dist
                            best_fire = (fx, fy)
                    
                    if best_fire:
                        robot.set_target(best_fire[0], best_fire[1], env_map)
                        # 注意：这里我们允许 detected_fires 被移除
                        if best_fire in detected_fires:
                            detected_fires.remove(best_fire)

            # 让机器人行动 (包含内部的资源检查与状态机跳转)
            robot.step(env_map)

        # --- 4. 渲染 ---
        screen.fill(COLOR_BG)
        draw_grid(screen, env_map)
        
        for drone in drones:
            drone.draw(screen)
        for robot in robots:
            robot.draw(screen)
            
        draw_sidebar(screen, env_map, robots)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()