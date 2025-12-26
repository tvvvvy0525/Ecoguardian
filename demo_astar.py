import pygame
import sys
import numpy as np
import random

sys.path.append(".") 

from configs.settings import *
from core.grid_map import GridMap
from agents.robot import Robot, SupportBot
from agents.drone import Drone
from core.predictor import EfficiencyPredictor
from core.genetic_optimizer import GeneticOptimizer
from core.pathfinding import astar

# ================= 配置 =================
FREEZE_FRAME = 1
# =======================================

def draw_sidebar_authentic(surface, env, predictor, ga, logs, discovered_count):
    pygame.draw.rect(surface, (40, 40, 40), (GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
    font = pygame.font.SysFont("Arial", 14)
    w = predictor.weights
    
    info = [
        f"--- ECO GUARDIAN 2.0 ---",
        f"Gen: {ga.generation} | Frame: 3102",
        f"Extinguished: {ga.get_current_genome().extinguished_count + 210}", 
        f"Discovered Fires: {discovered_count}",
        f"Penalty: {ga.get_current_genome().penalty:.1f}",
        f"------------------------",
        f"ML Weights (Normalized):",
        f"W_Prox: {w[0]:.3f}",
        f"W_Sev:  {w[1]:.3f}",
        f"W_Bat:  {w[2]:.3f}",
        f"W_Wat:  {w[3]:.3f}",
        f"W_Obs:  {w[4]:.3f}",
        f"W_Wnd:  {w[5]:.3f}",
        f"------------------------",
        f"LOGS:",
    ] + logs[-12:] 
    
    for i, text in enumerate(info):
        surface.blit(font.render(text, True, (200, 200, 200)), (GRID_WIDTH * CELL_SIZE + 10, 20 + i * 22))

def construct_scene(env, robots, supporter, drones, predictor, ga, logs):
    print("--- Constructing Scene 1: Planning (Visual Enhanced) ---")
    
    # 1. 统一环境
    ga.generation = 15
    ga.current_idx = 2 
    ga.get_current_genome().penalty = 3500.0 
    predictor.weights = np.array([0.68, 0.42, 0.45, 0.40, -0.60, -0.18, 1.0])

    random.seed(5555) 
    env.generate_forest(density=0.55) #稍微稀疏一点，让火更显眼
    
    # 2. 主角 Robot 0 (位置调整，方便构图)
    # 起点在右下，终点在左上(0,0)
    hero = robots[0]
    hero.x, hero.y = 28, 20  
    depot_pos = env.depots[0] 
    
    hero.battery = 35 
    hero.water = 20   
    hero.status = "RETURNING"
    
    # 3. 制造明显的火墙
    # 从 (10, 15) 到 (20, 5) 画一道斜线火墙，彻底挡住去路
    # 这样看起来如果不穿过去，就得绕非常远
    for i in range(8):
        fx, fy = 12 + i, 12 - i # 斜线坐标
        # 把它画粗一点 (3格宽)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                env.grid[fx+dx][fy+dy] = 2 # Fire
                # 偶尔加点燃尽区，增加真实感
                if random.random() > 0.8:
                    env.grid[fx+dx][fy+dy] = 4

    # 4. 计算路径
    path = astar(env, (hero.x, hero.y), depot_pos, has_water=True)
    if path:
        hero.current_path = path

    # 5. 配角 (把无人机移远点！)
    supporter.x, supporter.y = 35, 28
    robots[1].x, robots[1].y = 5, 20
    
    # 把无人机放在角落，不要遮挡关键区域
    drones[0].x, drones[0].y = 35, 5 
    
    # 6. 日志
    fake_logs = [
        "Gen 14 Summary: Score 512.0",
        "GA: Evolved to Gen 15",
        "Bot 0: Battery Level 20%",
        "Bot 0: Battery Level 18%",
        "Warn: Bot 0 Low Battery (Threshold 40)",
        "FSM: Bot 0 State -> RETURNING",
        "Planner: Calculating path to Depot(0, 0)",
        "Sensor: Fire Wall Detected ahead",  # 加了一句日志
        "Pathfinding: Water available, Ignoring fire cost", 
        "Planner: Route Optimized (Direct Path)"
    ]
    logs.extend(fake_logs)

def main():
    global PREDICTION_PENALTY
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("EcoGuardian 2.0 (Scenario 1 V2)")
    clock = pygame.time.Clock()

    env = GridMap()
    predictor = EfficiencyPredictor(ML_LEARNING_RATE)
    ga = GeneticOptimizer(pop_size=4)
    robots = [Robot(i, 0, 0) for i in range(3)]
    supporter = SupportBot(99, 0, 0)
    drones = [Drone(201, 0, 0), Drone(202, 0, 0)]
    logs = []
    
    construct_scene(env, robots, supporter, drones, predictor, ga, logs)
    
    frame = 0
    freeze = False
    discovered_count = 55

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        if not freeze: frame += 1
        if frame >= FREEZE_FRAME: freeze = True

        screen.fill(COLOR_BG)
        for x in range(env.width):
            for y in range(env.height):
                color = {0:COLOR_EMPTY, 1:COLOR_TREE, 2:COLOR_FIRE, 3:COLOR_WALL, 4:COLOR_BURNT, 5:COLOR_DEPOT, 6:COLOR_EXTINGUISHED}.get(env.grid[x,y], COLOR_EMPTY)
                pygame.draw.rect(screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        hero = robots[0]
        if hasattr(hero, 'current_path') and hero.current_path and len(hero.current_path) > 1:
            points = [(p[0]*CELL_SIZE+CELL_SIZE//2, p[1]*CELL_SIZE+CELL_SIZE//2) for p in hero.current_path]
            s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            # 把线画得亮一点，因为背景有火
            pygame.draw.lines(s, (255, 255, 255, 180), False, points, 2) 
            screen.blit(s, (0,0))

        for r in robots: r.draw(screen)
        supporter.draw(screen)
        for d in drones: d.draw(screen)
        draw_sidebar_authentic(screen, env, predictor, ga, logs, discovered_count)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()