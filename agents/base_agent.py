# agents\base_agent.py
import pygame
from configs.settings import *

class BaseAgent:
    def __init__(self, agent_id, x, y, color):
        """初始化基类代理"""
        self.id = agent_id # 智能体id
        self.x = x # 智能体坐标 x
        self.y = y # 智能体坐标 y
        self.color = color # 智能体颜色
        self.type = "Base" # 智能体类型

    def move(self, dx, dy, grid_width, grid_height):
        """尝试移动，如果出界则不移动"""
        nx, ny = self.x + dx, self.y + dy # 新的x坐标和y坐标
        if 0 <= nx < grid_width and 0 <= ny < grid_height: # 如果新的x坐标和y坐标在地图范围内
            self.x = nx # 更新x坐标
            self.y = ny # 更新y坐标
            return True # 返回True表示移动成功
        return False # 返回False表示移动失败

    def draw(self, surface):
        """在屏幕上绘制自己 (简单的方块或圆形)"""
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        padding = 2
        rect = pygame.Rect(px + padding, py + padding, CELL_SIZE - padding*2, CELL_SIZE - padding*2)
        pygame.draw.rect(surface, self.color, rect)