import pygame
import sys
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.settings import *
from core.grid_map import GridMap
from agents.robot import Robot, SupportBot
from agents.drone import Drone
from core.predictor import EfficiencyPredictor
from core.genetic_optimizer import GeneticOptimizer


class Logger(object):
    def __init__(self, filename="simulation.log"):
        self.terminal_out = sys.stdout
        self.terminal_err = sys.stderr
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal_out.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal_out.flush()
        self.terminal_err.flush()
        self.log.flush()


def log_system_status(frame, env, robots, predictor, ga, penalty):
    w = predictor.weights
    genome = ga.get_current_genome()
    idle_stat = getattr(genome, "idle_frames", 0)

    print(f"\n" + "=" * 50)
    print(f" FRAME: {frame} | GEN: {ga.generation} | INDIVIDUAL: {ga.current_idx + 1}")
    print("-" * 50)

    # 1. 环境状态
    active_fires = np.sum(env.grid == 2)
    extinguished = np.sum(env.grid == 6)
    print(f"[ENV] Active Fires: {active_fires} | Total Extinguished: {extinguished}")

    # 2. 机器人实时状态
    # 监控 Stranded 是为了检查是否有机器人因为贪婪抢单而死在半路
    stranded_count = sum(1 for r in robots if r.status == "STRANDED")
    idle_current = sum(1 for r in robots if r.status == "IDLE")
    print(
        f"[BOT] Idle: {idle_current} | Stranded: {stranded_count} | Moving: {len(robots)-idle_current-stranded_count}"
    )

    # 3. 遗传算法参数 (核心监控区)
    # Radius: 决定了避嫌范围 (越小越激进)
    # IdleSum: 决定了闲置惩罚力度 (如果你发现 Radius 很小但 IdleSum 很大，说明地图太大了或者火太少了)
    print(
        f"[GA ] Radius: {genome.radius} | Penalty: {penalty:.1f} | IdleSum (累计闲置): {idle_stat}"
    )

    # 4. 机器学习权重 (ML监控)
    # 检查 Bat/Wat 是否死守 0.3 底线，检查 Sev 是否过低
    print(f"[ML ] Weights Snapshot:")
    print(f"      Prox: {w[0]:.3f} | Sev: {w[1]:.3f} | Wind: {w[5]:.3f}")
    print(f"      Bat : {w[2]:.3f} | Wat: {w[3]:.3f} | Obs : {w[4]:.3f}")

    # 警告提示
    if w[2] <= 0.31 or w[3] <= 0.31:
        print("      ⚠️  WARNING: Resource weights near floor (Risk of Stranding)")

    print("=" * 50 + "\n")


def save_weight_chart(history, frame, generation, save_dir="plots"):
    """生成并保存当前的权重进化图"""
    if not history:
        return

    # 确保目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 准备数据
    data_np = np.array(history)
    x_axis = np.arange(len(history))

    # 设置绘图风格
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))  # 图片大小 10x6 英寸

    # 定义颜色和标签
    labels = ["Prox (Dist)", "Sev (Fire)", "Bat", "Wat", "Wind", "Obs"]
    indices = [0, 1, 2, 3, 5, 4]
    colors = ["#ff3333", "#ffaa00", "#00ff00", "#3399ff", "#00ffff", "#aa66ff"]

    # 绘制线条
    for i, idx in enumerate(indices):
        ax.plot(x_axis, data_np[:, i], label=labels[i], color=colors[i], linewidth=1.5)

    # 设置装饰
    ax.set_title(f"EcoGuardian ML Weights Evolution (Gen {generation} - Frame {frame})")
    ax.set_xlabel("Simulation Frames")
    ax.set_ylabel("Weight Value")
    ax.set_ylim(-0.8, 1.2)  # 固定 Y 轴范围
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    # 保存文件
    filename = f"{save_dir}/gen_{generation}_frame_{frame}.png"
    plt.savefig(filename, dpi=100)
    plt.close(fig)  # 关闭图表释放内存
    print(f"[System] 📸 Chart saved to {filename}")


# 计算指定位置周围的障碍物密度 (归一化输出 0~1)
def get_local_obs_density(grid_map, x, y):
    x1, x2 = max(0, x - 1), min(grid_map.width, x + 2)
    y1, y2 = max(0, y - 1), min(grid_map.height, y + 2)
    area = grid_map.grid[x1:x2, y1:y2]
    return np.sum(area == 3) / area.size


# 绘制侧边栏
def draw_sidebar(surface, env, predictor, ga, logs, discovered_count):
    pygame.draw.rect(
        surface, (40, 40, 40), (GRID_WIDTH * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    )
    font = pygame.font.SysFont("Arial", 14)
    info = [
        f"--- ECO GUARDIAN 2.0 ---",
        f"Gen: {ga.generation} | Frame: {ga.current_idx}",
        f"Extinguished: {np.sum(env.grid==6)}",
        f"Discovered Fires: {discovered_count}",
        f"Penalty: {PREDICTION_PENALTY:.1f}",
        f"------------------------",
        f"ML Weights (Normalized):",
        f"W_Prox: {predictor.weights[0]:.3f}",
        f"W_Sev:  {predictor.weights[1]:.3f}",
        f"W_Bat:  {predictor.weights[2]:.3f}",
        f"W_Wat:  {predictor.weights[3]:.3f}",
        f"W_Obs:  {predictor.weights[4]:.3f}",
        f"W_Wnd:  {predictor.weights[5]:.3f}",  # 显示风向权重
        f"------------------------",
        f"LOGS:",
    ] + logs[-10:]
    for i, text in enumerate(info):
        surface.blit(
            font.render(text, True, (200, 200, 200)),
            (GRID_WIDTH * CELL_SIZE + 10, 20 + i * 22),
        )


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # 全局变量声明
    global PREDICTION_PENALTY

    env = GridMap()

    # 初始点火
    for _ in range(3):
        env.ignite_random()

    discovered_fires = set()

    # 初始化 AI 模块
    predictor = EfficiencyPredictor(ML_LEARNING_RATE)
    ga = GeneticOptimizer(pop_size=4)
    current_penalty = ga.get_current_genome().penalty
    last_extinguished_total = 0  # 用于计算本周期内的灭火增量

    # 初始化 Agents
    robots = [Robot(i, env.depots[i % 4][0], env.depots[i % 4][1]) for i in range(3)]
    supporter = SupportBot(99, env.depots[0][0], env.depots[0][1])
    drones = [Drone(201, 10, 10), Drone(202, 30, 20)]

    frame, logs = 0, []
    weight_history = []  # 用于存储历史权重数据

    while True:
        frame += 1
        current_genome = ga.get_current_genome()
        if not hasattr(current_genome, "idle_frames"):
            current_genome.idle_frames = 0
        # --- 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                env.ignite_random()

        # --- 环境更新 ---
        if frame % 12 == 0:
            env.update_fire_spread()

        # --- 无人机感知循环 ---
        for drone in drones:
            drone.step(env, frame)
            new_reports = drone.scan(env, frame)
            for f_pos in new_reports:
                discovered_fires.add((int(f_pos[0]), int(f_pos[1])))

        # 清理已熄灭的火点
        discovered_fires = {
            f for f in discovered_fires if env.get_state(f[0], f[1]) == 2
        }

        # --- [核心逻辑] 任务调度 (Dispatcher) ---
        if frame % 20 == 0:
            # 1. 获取资源
            idle_robots = [r for r in robots if r.status == "IDLE"]
            active_fires = list(discovered_fires)

            if idle_robots and active_fires:
                # 按火势降序排列
                def get_fire_severity(pos):
                    fx, fy = pos
                    # 1. 计算安全的切片边界，防止负数索引导致的“穿墙”读取
                    x_min = max(0, fx - 1)
                    x_max = min(env.width, fx + 2)
                    y_min = max(0, fy - 1)
                    y_max = min(env.height, fy + 2)

                    # 2. 安全切片
                    area = env.grid[x_min:x_max, y_min:y_max]

                    # 3. 计算严重度 (依然除以 9.0 做归一化，保持特征尺度一致)
                    return np.sum(area == 2) / 9.0

                active_fires.sort(key=get_fire_severity, reverse=True)

                for f_pos in active_fires:
                    if not idle_robots:
                        break

                    is_crowded = False
                    for r in robots:
                        target = r.target if r.target else (r.x, r.y)
                        if (
                            abs(target[0] - f_pos[0]) + abs(target[1] - f_pos[1])
                            <= current_genome.radius
                        ):
                            is_crowded = True
                            break

                    if is_crowded:
                        continue

                    severity = get_fire_severity(f_pos)

                    # 竞价选拔
                    best_robot = None
                    min_cost = 999999
                    best_feat = None

                    for r in idle_robots:
                        dist_m = abs(r.x - f_pos[0]) + abs(r.y - f_pos[1])
                        vec_x = (f_pos[0] - r.x) / (dist_m if dist_m > 0 else 1)
                        vec_y = (f_pos[1] - r.y) / (dist_m if dist_m > 0 else 1)
                        wind_align = (
                            vec_x * env.wind_direction[0]
                            + vec_y * env.wind_direction[1]
                        )
                        max_map_dist = GRID_WIDTH + GRID_HEIGHT

                        feats = [
                            1.0 - (dist_m / max_map_dist),  # 使用动态地图尺寸
                            get_fire_severity(f_pos),  # 函数内部已归一化 (/9.0)
                            r.battery / ROBOT_MAX_BATTERY,  # 使用常量 (200)
                            r.water / ROBOT_MAX_WATER,  # 使用常量 (30)
                            get_local_obs_density(
                                env, f_pos[0], f_pos[1]
                            ),  # 函数内部已归一化
                            wind_align,  # 自然归一化 (-1~1)
                        ]

                        cost = r.calculate_bid(
                            f_pos,
                            feats,
                            predictor,
                            current_genome.penalty,
                        )

                        if cost < min_cost:
                            min_cost = cost
                            best_robot = r
                            best_feat = feats

                    # 派遣逻辑
                    if best_robot and min_cost < BID_REJECT_THRESHOLD:
                        if best_robot.set_target(f_pos[0], f_pos[1], env, best_feat):
                            idle_robots.remove(best_robot)
                            msg = f"Dispatch: Fire {f_pos} -> Bot {best_robot.id}"
                            logs.append(msg)
                            print(msg)  # [恢复控制台日志]
        # grid_before = env.grid.copy()
        # --- 执行 Agent 更新 ---
        for r in robots:
            r.step(env, predictor, robots, current_genome=current_genome)
        supporter.step(env, robots)
        idle_count = sum(1 for r in robots if r.status == "IDLE")
        current_genome.idle_frames += idle_count
        indices_to_plot = [0, 1, 2, 3, 5, 4]  # Prox, Sev, Bat, Wat, Wind, Obs
        current_weights = [predictor.weights[i] for i in indices_to_plot]
        weight_history.append(current_weights)
        # --- 遗传算法进化 ---
        if frame % GA_EVOLVE_INTERVAL == 0:
            current_total = np.sum(env.grid == 6)
            current_genome = ga.get_current_genome()
            current_genome.extinguished_count = current_total - last_extinguished_total
            last_extinguished_total = current_total
            current_genome.stranded_count = sum(
                1 for r in robots if r.status == "STRANDED"
            )
            log_system_status(frame, env, robots, predictor, ga, current_penalty)
            print(
                f"[GA Eval] Gen {ga.generation}: Ext:{current_genome.extinguished_count}, "
                f"SevBonus:{current_genome.severity_bonus:.1f}, Stranded:{current_genome.stranded_count}"
            )
            save_weight_chart(weight_history, frame, ga.generation)
            ga.next_step()
            current_penalty = ga.get_current_genome().penalty

        # --- 渲染画面 ---
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
            d.draw(screen)
        supporter.draw(screen)
        draw_sidebar(screen, env, predictor, ga, logs, len(discovered_fires))
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    # 劫持所有输出到日志文件
    logger = Logger("simulation.log")
    sys.stdout = logger
    sys.stderr = logger
    print("--- Simulation Started: Logging to simulation.log ---")
    main()
