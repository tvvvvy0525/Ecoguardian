import heapq


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


# [清单1] 更新 astar
def astar(grid_map, start, end, has_water=True):
    if grid_map.get_state(end[0], end[1]) == 3:
        return None
    start_node, end_node = Node(None, start), Node(None, end)
    open_list, closed_list = [], []
    heapq.heappush(open_list, start_node)

    max_iterations = (grid_map.width * grid_map.height) * 2
    iter_count = 0
    while len(open_list) > 0 and iter_count < max_iterations:
        iter_count += 1
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
            )
            if not (0 <= nx < grid_map.width and 0 <= ny < grid_map.height):
                continue

            cell_val = grid_map.get_state(nx, ny)
            if cell_val == 3:
                continue

            # [清单1] 代价逻辑
            move_cost = 1
            if cell_val == 2:
                move_cost = 1 if has_water else 50  # 没水时设为极高代价

            new_node = Node(current_node, (nx, ny))
            if new_node in closed_list:
                continue

            new_node.g = current_node.g + move_cost
            new_node.h = abs(nx - end[0]) + abs(ny - end[1])
            new_node.f = new_node.g + new_node.h

            if any(
                new_node == open_n and new_node.g > open_n.g for open_n in open_list
            ):
                continue
            heapq.heappush(open_list, new_node)
    return None
