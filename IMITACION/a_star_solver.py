# a_star_solver.py
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid_size, agent_pos, goal_pos, poisons):
    start = tuple(agent_pos)
    goal = tuple(goal_pos)
    
    close_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    open_heap = [(f_score[start], start)]

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.insert(0, current)
            path.pop(0)
            return path

        close_set.add(current)
        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + i, current[1] + j)
            
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            if neighbor in close_set or any(tuple(p) == neighbor for p in poisons):
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return None