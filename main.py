import pygame
import sys
import numpy as np
import math
from configs.settings import *
from core.grid_map import GridMap
from agents.drone import Drone
from agents.robot import Robot
from core.predictor import EfficiencyPredictor

# 初始化 Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("EcoGuardian - Intelligent Utility Evaluator")
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

def draw_sidebar(surface, grid_map, robots, logs, predictor):
    sidebar_rect = pygame.Rect(GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, (40, 40, 40), sidebar_rect)
    font = pygame.font.SysFont("Arial", 16)
    
    # 统计
    trees = np.sum(grid_map.grid == 1)
    fires = np.sum(grid_map.grid == 2)
    saved_by_bot = np.sum(grid_map.grid == 6)
    
    # 显示权重
    w_diff = predictor.weights[0]
    w_size = predictor.weights[1]
    w_bias = predictor.weights[2]
    
    texts = [
        f"EcoGuardian ML",
        f"----------------",
        f"Samples: {predictor.training_count}",
        f"W_Diff: {w_diff:.3f}",
        f"W_Size: {w_size:.3f}",
        f"Bias: {w_bias:.3f}",
        f"----------------",
        f"Trees: {trees}",
        f"Fires: {fires} (!)",
        f"Extinguished: {saved_by_bot}", 
        f"----------------",
        f"Robots Total: {len(robots)}",
        f"----------------",
        f"KEYS:",
        f"[SPACE]: Ignite",
        f"[R]: Reset"
    ]
    
    current_y = 20
    line_height = 25

    for text in texts:
        color = (200, 200, 200)
        if "W_Diff" in text: color = (100, 255, 100) 
        if "W_Size" in text: color = (100, 200, 255)
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (GRID_WIDTH * CELL_SIZE + 10, current_y))
        current_y += line_height

    start_y = current_y + 20 
    title_surf = pygame.font.SysFont("Arial", 16).render("--- Mission Log ---", True, (200, 200, 200))
    surface.blit(title_surf, (GRID_WIDTH * CELL_SIZE + 10, start_y))
    
    font_small = pygame.font.SysFont("Arial", 14)
    log_start_y = start_y + 25 
    
    for i, msg in enumerate(logs):
        msg_surf = font_small.render(msg, True, (255, 215, 0)) 
        surface.blit(msg_surf, (GRID_WIDTH * CELL_SIZE + 10, log_start_y + i * 20))

def get_fire_cluster_size(grid_map, fx, fy):
    """简单计算火团大小 (BFS)"""
    count = 0
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx, ny = fx+dx, fy+dy
            if grid_map.get_state(nx, ny) == 2:
                count += 1
    return count

def main():
    cmd_logs = []
    env_map = GridMap()
    env_map.ignite_random() 

    # --- 初始化智能体 ---
    drones = [Drone(0, 0, 0), Drone(1, GRID_WIDTH-1, GRID_HEIGHT-1)]
    robots = [
        Robot(101, 2, GRID_HEIGHT-2),
        Robot(102, 5, GRID_HEIGHT-2),
        Robot(103, GRID_WIDTH-2, GRID_HEIGHT-2)
    ]
    
    # 初始化预测器
    predictor = EfficiencyPredictor(learning_rate=ML_LEARNING_RATE)
    
    detected_fires = set()
    unreachable_tasks = {}
    
    frame_count = 0 

    running = True
    while running:
        frame_count += 1 
        
        # --- 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env_map = GridMap()
                    detected_fires.clear()
                    unreachable_tasks.clear() 
                    robots[0].x, robots[0].y = 2, GRID_HEIGHT-2
                    robots[1].x, robots[1].y = 5, GRID_HEIGHT-2
                    robots[2].x, robots[2].y = GRID_WIDTH-2, GRID_HEIGHT-2
                    for r in robots:
                        r.status = "IDLE"
                        r.target = None
                        r.battery = ROBOT_MAX_BATTERY
                        r.water = ROBOT_MAX_WATER
                    predictor.weights = np.array([0.0, 0.0, 0.0])
                elif event.key == pygame.K_SPACE:
                    env_map.ignite_random()

        # --- 智能体逻辑更新 ---
        if frame_count % 5 == 0 and env_map:
            env_map.update_fire_spread()
            
        current_frame_fires = set()
        for drone in drones:
            drone.step(env_map, frame_count) 
            found = drone.scan(env_map, frame_count)
            current_frame_fires.update(found)
        for robot in robots:
            local_found = robot.scan_local(env_map)
            current_frame_fires.update(local_found)
        detected_fires.update(current_frame_fires)
        detected_fires = {f for f in detected_fires if env_map.get_state(f[0], f[1]) == 2}
        
        # --- 核心调度 (修复互斥锁) ---
        if frame_count % 10 == 0:
            
            # 清理黑名单
            expired = [k for k, v in unreachable_tasks.items() if v < frame_count]
            for k in expired:
                del unreachable_tasks[k]
            
            # 1. 获取已经在干活的机器人的目标
            busy_robots = [r for r in robots if r.target]
            
            # [新增] 本帧内新分配的目标 (Frame Mutex Lock)
            # 用于防止 Robot A 选了火点 X 后，Robot B 在同一帧内也选 X
            frame_locked_fires = set()

            for robot in robots:
                if robot.status == "IDLE" and detected_fires:
                    if robot.battery < ROBOT_LOW_BATTERY_THRESHOLD or robot.water <= 0:
                        continue 
                    
                    all_candidates = [] 
                    
                    for fire_pos in detected_fires:
                        # 检查1: 是否被老员工锁定 (Previous Frame)
                        is_locked_prev = False
                        for r in busy_robots:
                            if r.target == fire_pos:
                                is_locked_prev = True
                                break
                        if is_locked_prev: continue
                        
                        # [新增] 检查2: 是否被新员工锁定 (Current Frame)
                        if fire_pos in frame_locked_fires:
                            continue
                        
                        if (robot.id, fire_pos) in unreachable_tasks:
                            continue

                        # --- ML 特征提取 ---
                        my_dist = abs(robot.x - fire_pos[0]) + abs(robot.y - fire_pos[1])
                        
                        closest_ally_dist = 999
                        for ally in robots:
                            if ally.id == robot.id: continue
                            d = abs(ally.x - fire_pos[0]) + abs(ally.y - fire_pos[1])
                            
                            # 判定谁是竞争者：
                            # 1. 已经有目标的队友
                            if ally.target:
                                t_dist = abs(ally.target[0] - fire_pos[0]) + abs(ally.target[1] - fire_pos[1])
                                if t_dist < 5:
                                    if d < closest_ally_dist: closest_ally_dist = d
                            # 2. [新增] 本帧刚接了单的队友
                            # 我们的 frame_locked_fires 只是火点集合，如果要更精确，可以记录 (bot, fire)
                            # 这里简化处理：如果队友在附近且火点被锁了，那就算竞争
                            # 但实际上我们在上面已经 continue 掉了 locked fires，所以这里不用太担心
                            pass

                        
                        if closest_ally_dist == 999: closest_ally_dist = 50 
                        
                        diff_dist = my_dist - closest_ally_dist
                        fire_size = get_fire_cluster_size(env_map, fire_pos[0], fire_pos[1])
                        
                        current_features = (diff_dist, fire_size)
                        cost = robot.calculate_bid(fire_pos, current_features, predictor)
                        
                        all_candidates.append((cost, fire_pos, current_features))
                    
                    if not all_candidates:
                        continue
                        
                    # 优选 -> 兜底
                    good_candidates = [c for c in all_candidates if c[0] <= BID_REJECT_THRESHOLD]
                    good_candidates.sort(key=lambda x: x[0])
                    
                    final_candidates = []
                    is_panic = False
                    
                    if good_candidates:
                        final_candidates = good_candidates
                    else:
                        all_candidates.sort(key=lambda x: x[0])
                        final_candidates = all_candidates
                        is_panic = True
                    
                    task_assigned = False
                    for cost, fire_pos, features in final_candidates:
                        # 最后再检查一遍锁 (虽然上面查过了，但为了保险)
                        if fire_pos in frame_locked_fires:
                            continue

                        success = robot.set_target(fire_pos[0], fire_pos[1], env_map, features=features)
                        
                        if success:
                            if not is_panic:
                                log_msg = f"> Assign: Bot {robot.id} -> ({fire_pos[0]},{fire_pos[1]})"
                                cmd_logs.append(log_msg)
                            
                            task_assigned = True
                            # [关键] 立即加锁！
                            frame_locked_fires.add(fire_pos)
                            break 
                        else:
                            unreachable_tasks[(robot.id, fire_pos)] = frame_count + 50
                            
                    if len(cmd_logs) > 8: 
                        cmd_logs.pop(0)

        # --- 行动 ---
        for robot in robots:
            robot.step(env_map, predictor)

        # --- 渲染 ---
        screen.fill(COLOR_BG)
        draw_grid(screen, env_map)
        for drone in drones:
            drone.draw(screen)
        for robot in robots:
            robot.draw(screen)
        
        draw_sidebar(screen, env_map, robots, cmd_logs, predictor)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()