# core/genetic_optimizer.py
import random
import numpy as np


class Genome:
    def __init__(self, penalty=None, radius=None):
        # 基因 1: 拥挤惩罚权重 (0 ~ 1000)
        self.penalty = penalty if penalty is not None else random.uniform(1000, 4000)
        # 基因 2: 拥挤半径 (1 ~ 8)
        self.radius = radius if radius is not None else random.randint(2, 6)

        # 适应度统计
        self.fitness = 0
        self.extinguished_count = 0
        self.severity_bonus = 0      # [新增] 灭大火的额外奖励
        self.stranded_count = 0      # [新增] 搁浅次数惩罚
        self.crowded_frames = 0  # 记录发生拥挤的帧数
        self.idle_frames = 0  # [新增] 记录闲置总帧数

    def mutate(self, rate=0.1):
        """随机变异"""
        if random.random() < rate:
            self.penalty += random.uniform(-500, 500)
            self.penalty = max(0, min(4000, self.penalty))
        if random.random() < rate:
            self.radius += random.choice([-1, 1])
            self.radius = max(2, min(4, self.radius))


class GeneticOptimizer:
    def __init__(self, pop_size=4):
        self.pop_size = pop_size
        self.population = [Genome() for _ in range(pop_size)]
        self.current_idx = 0
        self.generation = 1

        # 确保种群里有一个较激进的初始值作为种子
        self.population[0] = Genome(penalty=2500, radius=5)
    def evaluate_fitness(self, genome):
        """
        重构评价函数：
        Score = (基础灭火 * 10) + (火场强度奖 * 5) - (搁浅惩罚 * 100) - (拥挤惩罚 * 10)
        """
        score = (genome.extinguished_count * 15.0) + \
                (genome.severity_bonus * 5.0) - \
                (genome.stranded_count * 150.0) - \
                (genome.crowded_frames * 10.0) - \
                (genome.idle_frames * 8.0)
        
        genome.fitness = max(1, score) # 确保分数为正
        return genome.fitness

    def get_current_genome(self):
        """获取当前基因组"""
        return self.population[self.current_idx]

    def record_success(self):
        """记录一次成功灭火"""
        self.population[self.current_idx].extinguished_count += 1

    def record_crowding(self):
        """记录一次拥挤事件"""
        self.population[self.current_idx].crowded_frames += 1

    def evaluate_fitness(self, genome):
        """计算适应度: 灭火奖励 - 拥挤惩罚"""
        # 逻辑: 我们想要多灭火，且少扎堆
        score = (genome.extinguished_count * 20.0) - (genome.crowded_frames * 15.0)
        genome.fitness = score
        return score

    def next_step(self):
        """切换到下一个个体，如果一轮结束则进化"""
        # 1. 结算当前个体的适应度
        current = self.population[self.current_idx]
        self.evaluate_fitness(current)

        # 2. 移动索引
        self.current_idx += 1

        # 3. 如果一代跑完了，进行进化
        if self.current_idx >= self.pop_size:
            self.evolve()
            self.current_idx = 0
            self.generation += 1
            return True  # 表示开启了新的一代
        return False

    def evolve(self):
        """进化逻辑: 精英保留 + 变异"""
        # 按适应度排序 (高的在前)
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        best = self.population[0]
        print(f"--- Generation {self.generation} Complete ---")
        print(
            f"Best Genome: Penalty={best.penalty:.1f}, Radius={best.radius}, Score={best.fitness:.1f}"
        )

        # 精英策略: 保留最好的 1 个，剩下的由它变异产生
        new_pop = [best]  # 保留冠军

        while len(new_pop) < self.pop_size:
            # 复制冠军并变异
            child = Genome(best.penalty, best.radius)
            child.mutate(rate=0.5)  # 较高的变异率以探索
            # 重置统计数据
            child.extinguished_count = 0
            child.crowded_frames = 0
            new_pop.append(child)

        # 重置冠军的统计数据以便下一轮测试
        best.extinguished_count = 0
        best.crowded_frames = 0

        self.population = new_pop
