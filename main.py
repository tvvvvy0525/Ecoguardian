import pygame, sys, numpy as np
from configs.settings import *
from core.grid_map import GridMap
from agents.robot import Robot, SupportBot
from agents.drone import Drone
from core.predictor import EfficiencyPredictor
from core.genetic_optimizer import GeneticOptimizer


def get_local_obs_density(grid_map, x, y):
    x1, x2 = max(0, x - 1), min(grid_map.width, x + 2)
    y1, y2 = max(0, y - 1), min(grid_map.height, y + 2)
    area = grid_map.grid[x1:x2, y1:y2]
    return np.sum(area == 3) / area.size


def draw_sidebar(surface, env, predictor, ga, logs, discovered_count):
    pygame.draw.rect(
        surface, (40, 40, 40), (GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    )
    font = pygame.font.SysFont("Arial", 14)
    info = [
        f"--- ECO GUARDIAN 2.0 ---",
        f"Gen: {ga.generation} | Frame: {ga.current_idx}",
        f"Extinguished: {np.sum(env.grid==6)}",
        f"Discovered Fires: {discovered_count}",  # 无人机发现的数量
        f"Penalty: {PREDICTION_PENALTY:.1f}",
        f"------------------------",
        f"ML Weights (5D):",
        f"W_Dist: {predictor.weights[0]:.3f}",
        f"W_Size: {predictor.weights[1]:.3f}",
        f"W_Bat: {predictor.weights[2]:.3f}",
        f"W_Wat: {predictor.weights[3]:.3f}",
        f"W_Obs: {predictor.weights[4]:.3f}",
        f"------------------------",
        f"LOGS:",
    ] + logs[-6:]
    for i, text in enumerate(info):
        surface.blit(
            font.render(text, True, (200, 200, 200)),
            (GRID_WIDTH * CELL_SIZE + 10, 20 + i * 22),
        )


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    env = GridMap()

    # 初始火点
    for _ in range(3):
        env.ignite_random()

    # [核心] 共享感知注册表：不再使用上帝视角
    discovered_fires = set()

    predictor = EfficiencyPredictor(ML_LEARNING_RATE)
    ga = GeneticOptimizer(pop_size=4)
    robots = [Robot(i, env.depots[i % 4][0], env.depots[i % 4][1]) for i in range(3)]
    supporter = SupportBot(99, env.depots[0][0], env.depots[0][1])
    drones = [Drone(201, 10, 10), Drone(202, 30, 20)]  # 无人机集群

    frame, logs = 0, []

    while True:
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                env.ignite_random()

        if frame % 12 == 0:
            env.update_fire_spread()

        # 1. 无人机巡逻与上报（交互核心）
        for drone in drones:
            drone.step(env, frame)
            new_reports = drone.scan(env, frame)
            for f_pos in new_reports:
                discovered_fires.add((int(f_pos[0]), int(f_pos[1])))

        # 2. 清理感知池（移除已熄灭的火点）
        discovered_fires = {
            f for f in discovered_fires if env.get_state(f[0], f[1]) == 2
        }

        # 3. 任务调度（仅基于无人机发现的信息）
        if frame % 20 == 0:
            for r in [rob for rob in robots if rob.status == "IDLE"]:
                if not discovered_fires:
                    break
                best_f, min_c, best_feat = None, 2000, None

                for f_pos in discovered_fires:
                    fx, fy = f_pos
                    if any(other.target == (fx, fy) for other in robots):
                        continue

                    feats = [
                        abs(r.x - fx) - 10,
                        np.sum(env.grid[fx - 1 : fx + 2, fy - 1 : fy + 2] == 2),
                        r.battery / ROBOT_MAX_BATTERY,
                        r.water / ROBOT_MAX_WATER,
                        get_local_obs_density(env, fx, fy),
                    ]
                    cost = r.calculate_bid((fx, fy), feats, predictor)
                    if cost < min_c:
                        min_c, best_f, best_feat = cost, (fx, fy), feats

                if best_f and min_c < BID_REJECT_THRESHOLD:
                    if r.set_target(best_f[0], best_f[1], env, best_feat):
                        logs.append(f"Coord: Bot {r.id} -> Fire {best_f}")

        # 4. 执行更新
        for r in robots:
            r.step(env, predictor)
        supporter.step(env, robots)

        # 进化与渲染
        if frame % GA_EVOLVE_INTERVAL == 0:
            ga.get_current_genome().extinguished_count = np.sum(env.grid == 6)
            ga.next_step()
            global PREDICTION_PENALTY
            PREDICTION_PENALTY = ga.get_current_genome().penalty

        screen.fill(COLOR_BG)
        for x in range(env.width):
            for y in range(env.height):
                color = {
                    0: COLOR_EMPTY,
                    1: COLOR_TREE,
                    2: COLOR_FIRE,
                    3: COLOR_WALL,
                    4: COLOR_BURNT,
                    5: COLOR_DEPOT,
                    6: COLOR_EXTINGUISHED,
                }.get(env.grid[x, y])
                pygame.draw.rect(
                    screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

        for r in robots:
            r.draw(screen)
        for d in drones:
            d.draw(screen)  # [修复] 渲染无人机
        supporter.draw(screen)
        draw_sidebar(screen, env, predictor, ga, logs, len(discovered_fires))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
