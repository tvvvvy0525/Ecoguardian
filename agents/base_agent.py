# agents\base_agent.py
import pygame
from configs.settings import CELL_SIZE

class BaseAgent:
    def __init__(self, agent_id, x, y, color):
        self.id = agent_id
        self.x = x
        self.y = y
        self.color = color
        self.type = "Base"

    def move(self, dx, dy, grid_width, grid_height):
        """尝试移动，如果出界则不移动"""
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            self.x = nx
            self.y = ny
            return True
        return False

    def draw(self, surface):
        """在屏幕上绘制自己 (简单的方块或圆形)"""
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        padding = 2
        rect = pygame.Rect(px + padding, py + padding, CELL_SIZE - padding*2, CELL_SIZE - padding*2)
        pygame.draw.rect(surface, self.color, rect)