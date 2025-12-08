# core/pathfinding.py
import heapq
import numpy as np

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # 离起点的距离
        self.h = 0  # 离终点的预估距离 (启发式)
        self.f = 0  # 总代价

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def astar(grid_map, start, end):
    """
    A* 寻路算法
    :param grid_map: GridMap 对象
    :param start: (x, y) 元组
    :param end: (x, y) 元组
    :return: 路径列表 [(x,y), (x,y)...] 或 None
    """
    
    # 如果终点是障碍物，无法到达
    end_val = grid_map.get_state(end[0], end[1])
    # 注意：这里我们允许终点是 Fire (2)，因为机器人要过去灭火
    # 但不允许终点是 Wall (3)
    if end_val == 3:
        return None

    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_list = []

    heapq.heappush(open_list, start_node)

    # 避免死循环的计数器
    max_iterations = (grid_map.width * grid_map.height) * 2
    iter_count = 0

    while len(open_list) > 0:
        iter_count += 1
        if iter_count > max_iterations:
            return None # 路径太长或无法到达

        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # 找到终点
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # 反转路径

        # 生成子节点 (上下左右)
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            # 检查边界
            if node_position[0] > (grid_map.width - 1) or node_position[0] < 0 or \
               node_position[1] > (grid_map.height - 1) or node_position[1] < 0:
                continue

            # 检查障碍物 (3: Wall)
            # 策略：如果不是 Wall，都可以走。
            # 也可以设定机器人不能走 Fire (2)，视难度而定。这里暂定避开 Fire 和 Wall
            cell_val = grid_map.get_state(node_position[0], node_position[1])
            if cell_val == 3 or (cell_val == 2 and node_position != end): 
                # 是墙，或者是火（且不是终点），则视为障碍
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child in closed_list:
                continue

            child.g = current_node.g + 1
            # 曼哈顿距离
            child.h = abs(child.position[0] - end_node.position[0]) + \
                      abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            # 检查 open_list 中是否有更优路径
            add_to_open = True
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    add_to_open = False
                    break
            
            if add_to_open:
                heapq.heappush(open_list, child)

    return None