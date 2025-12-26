import heapq

# Node 类保持不变
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

def astar(grid_map, start, end, has_water=True):
    # 1. 终点如果是墙，直接返回不可达
    if grid_map.get_state(end[0], end[1]) == 3:
        return None

    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    heapq.heappush(open_list, start_node)
    
    # [优化 1] 使用 g_costs 字典代替 closed_set
    # 键是坐标 (x,y)，值是到达该点的最小 G 值 (代价)
    # 作用：既能充当 visited 集合，又能进行“更优路径剪枝”
    g_costs = {start: 0} 

    while len(open_list) > 0:
        current_node = heapq.heappop(open_list)
        current_pos = current_node.position

        # [优化 2] 延迟删除 (Lazy Deletion)
        # 如果当前取出的节点代价比我们记录的最小代价还要大，说明这是个“过时”的垃圾节点，直接跳过
        if current_node.g > g_costs.get(current_pos, float('inf')):
            continue

        # 找到终点
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # 遍历邻居
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = (
                current_pos[0] + new_position[0],
                current_pos[1] + new_position[1],
            )

            # --- 基础检查 ---
            # 1. 边界检查
            if not (0 <= nx < grid_map.width and 0 <= ny < grid_map.height):
                continue
            # 2. 障碍物检查
            cell_val = grid_map.get_state(nx, ny)
            if cell_val == 3:
                continue

            # --- 代价计算 ---
            move_cost = 1
            if cell_val == 2:
                move_cost = 1 if has_water else 50
            
            new_g = current_node.g + move_cost

            # --- [核心优化逻辑] ---
            # 只有当我们找到了一条通往 (nx, ny) 的更短路径时，才处理它
            # 如果我们之前已经用更少的代价到达过这里，就不要再把这个邻居加入队列了
            if (nx, ny) not in g_costs or new_g < g_costs[(nx, ny)]:
                # 更新记分板
                g_costs[(nx, ny)] = new_g
                
                # 创建新节点并加入队列
                new_node = Node(current_node, (nx, ny))
                new_node.g = new_g
                new_node.h = abs(nx - end[0]) + abs(ny - end[1])
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_list, new_node)

    return None