import pygame
import sys
import numpy as np
from configs.settings import *
from core.grid_map import GridMap
from agents.drone import Drone
from agents.robot import Robot
# from agents.dispatcher import Dispatcher # 暂时不需要这个，我们在主循环直接调度

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
    5: COLOR_DEPOT,
    6: COLOR_EXTINGUISHED
}

def draw_grid(surface, grid_map):
    for x in range(grid_map.width):
        for y in range(grid_map.height):
            state = grid_map.grid[x][y]
            color = COLOR_MAP.get(state, COLOR_EMPTY)
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)

def draw_sidebar(surface, grid_map, robots, logs):
    """侧边栏显示更多信息"""
    sidebar_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, (40, 40, 40), sidebar_rect)
    font = pygame.font.SysFont("Arial", 16)
    
    # 统计
    trees = np.sum(grid_map.grid == 1)
    fires = np.sum(grid_map.grid == 2)
    burnt_natural = np.sum(grid_map.grid == 4)     # 自然烧完
    saved_by_bot = np.sum(grid_map.grid == 6)      # 机器人扑灭
    
    # 计算空闲机器人
    idle_bots = sum(1 for r in robots if r.status == "IDLE")
    
    texts = [
        f"EcoGuardian System",
        f"----------------",
        f"Wind: {grid_map.wind_name}",
        f"Strength: {WIND_STRENGTH}x",
        f"----------------",
        f"Trees: {trees}",
        f"Fires: {fires} (!)",
        f"Burnt (Natural): {burnt_natural}",
        f"Extinguished: {saved_by_bot}", 
        f"Efficiency: {saved_by_bot / (burnt_natural + saved_by_bot + 0.1) * 100:.1f}%",
        f"----------------",
        f"Robots Total: {len(robots)}",
        f"Robots Idle: {idle_bots}",
        f"----------------",
        f"KEYS:",
        f"[SPACE]: Ignite",
        f"[R]: Reset"
    ]
    
    current_y = 20 # 追踪当前的 Y 坐标
    line_height = 25

    for text in texts:
        color = (255, 100, 100) if "(!)" in text and fires > 0 else (200, 200, 200)
        if "Wind:" in text: color = (100, 200, 255) 
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (GRID_WIDTH * CELL_SIZE + 10, current_y))
        current_y += line_height # 每画一行，Y 坐标增加

    # --- 修复重叠的关键点 ---
    # 不要硬编码 350，而是在 current_y 的基础上加一点间距 (padding)
    start_y = current_y + 20 
    
    title_surf = pygame.font.SysFont("Arial", 16).render("--- Mission Log ---", True, (200, 200, 200))
    surface.blit(title_surf, (GRID_WIDTH * CELL_SIZE + 10, start_y))
    
    font_small = pygame.font.SysFont("Arial", 14)
    # 日志内容从标题下方开始
    log_start_y = start_y + 25 
    
    for i, msg in enumerate(logs):
        msg_surf = font_small.render(msg, True, (255, 215, 0)) 
        surface.blit(msg_surf, (GRID_WIDTH * CELL_SIZE + 10, log_start_y + i * 20))

def main():
    cmd_logs = [] # 命令日志
    env_map = GridMap()
    env_map.ignite_random() 

    # --- 1. 初始化智能体 ---
    drones = [
        Drone(0, 0, 0),
        Drone(1, GRID_WIDTH-1, GRID_HEIGHT-1) 
    ]
    
    robots = [
        Robot(101, 2, GRID_HEIGHT-2),
        Robot(102, 5, GRID_HEIGHT-2),
        Robot(103, GRID_WIDTH-2, GRID_HEIGHT-2)
    ]
    
    # dispatcher = Dispatcher() # 禁用外部调度器，使用主循环逻辑
    
    # 任务队列 (Set 防止重复)
    detected_fires = set()
    frame_count = 0 

    running = True
    while running:
        frame_count += 1 
        # --- 2. 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env_map = GridMap()
                    detected_fires.clear()
                    # 重置机器人位置
                    robots[0].x, robots[0].y = 2, GRID_HEIGHT-2
                    robots[1].x, robots[1].y = 5, GRID_HEIGHT-2
                    robots[2].x, robots[2].y = GRID_WIDTH-2, GRID_HEIGHT-2
                    for r in robots:
                        r.status = "IDLE"
                        r.target = None
                        r.battery = ROBOT_MAX_BATTERY
                        r.water = ROBOT_MAX_WATER
                        
                elif event.key == pygame.K_SPACE:
                    env_map.ignite_random()

        # --- 3. 智能体逻辑更新 ---
        
        if frame_count % 5 == 0 and env_map:
            env_map.update_fire_spread()
            
        # B. 无人机感知
        current_frame_fires = set()
        for drone in drones:
            drone.step(env_map, frame_count) 
            found = drone.scan(env_map, frame_count)
            for f in found:
                current_frame_fires.add(f)
        
        # 机器人也能看到脚边的火
        for robot in robots:
            local_found = robot.scan_local(env_map)
            for f in local_found:
                current_frame_fires.add(f)
        
        detected_fires.update(current_frame_fires)
        
        # 移除已经不再燃烧的火点 (清洗任务池)
        detected_fires = {f for f in detected_fires if env_map.get_state(f[0], f[1]) == 2}
        
        # --- C. 主循环核心调度 (Global Scheduler) ---
        # 每10帧执行一次分配，避免过于频繁抖动
        if frame_count % 10 == 0:
            
            # [关键步骤 1] 建立任务锁
            # 统计所有正在忙碌的机器人正在去哪里，这些火点不能分给别人
            locked_fires = set()
            for r in robots:
                # 注意：RETURNING 的目标是补给站，不算占用了火点
                if r.status in ["MOVING", "EXTINGUISHING"] and r.target:
                    locked_fires.add(r.target)
            
            # [关键步骤 2] 为每个空闲机器人分配任务
            for robot in robots:
                if robot.status == "IDLE" and detected_fires:
                    
                    # 资源检查：没水没电的不分配任务，让它自己在 step() 里触发返航
                    if robot.battery < ROBOT_LOW_BATTERY_THRESHOLD or robot.water <= 0:
                        continue 
                    
                    best_fire = None
                    min_cost = float('inf')
                    
                    # 遍历所有已知火点
                    for fire_pos in detected_fires:
                        
                        # [关键步骤 3] 排他性检查
                        # 如果这个火点已经被别人锁定了，完全无视它
                        if fire_pos in locked_fires:
                            continue
                            
                        # 计算代价 (距离 + 电池消耗权重)
                        cost = robot.calculate_bid(fire_pos)
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_fire = fire_pos
                    
                    # 如果找到了一个没人抢的火点
                    if best_fire:
                        robot.set_target(best_fire[0], best_fire[1], env_map)
                        
                        # [关键步骤 4] 立即加锁！
                        # 这样循环里的下一个机器人就不会选这个火点了
                        locked_fires.add(best_fire)
                        
                        # 记录日志
                        log_msg = f"> Assign: Bot {robot.id} -> ({best_fire[0]},{best_fire[1]})"
                        cmd_logs.append(log_msg)
                        if len(cmd_logs) > 8: 
                            cmd_logs.pop(0)

        # --- D. 让所有机器人行动 (必须每一帧都执行) ---
        for robot in robots:
            robot.step(env_map)

        # --- 4. 渲染 ---
        screen.fill(COLOR_BG)
        draw_grid(screen, env_map)
        
        for drone in drones:
            drone.draw(screen)
        for robot in robots:
            robot.draw(screen)
            
        draw_sidebar(screen, env_map, robots, cmd_logs)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()