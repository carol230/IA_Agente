# a_star_solver.py
"""
Implementación del algoritmo A* para navegación óptima en grid.

Este módulo proporciona funcionalidades para encontrar el camino más corto
entre dos puntos en una cuadrícula, evitando obstáculos (venenos). Utiliza
el algoritmo A* con distancia Manhattan como heurística para garantizar
optimalidad en entornos de grid.

Funciones:
    heuristic(a, b): Calcula distancia Manhattan entre dos puntos
    a_star_search(grid_size, agent_pos, goal_pos, poisons): Encuentra camino óptimo
"""
import heapq

def heuristic(a, b):
    """
    Calcula la distancia heurística entre dos puntos usando distancia Manhattan.
    
    La distancia Manhattan es la suma de las diferencias absolutas de sus
    coordenadas cartesianas. Es una heurística admisible para movimiento
    en grid con 4 direcciones, garantizando optimalidad en A*.
    
    Args:
        a (tuple): Coordenadas (x, y) del primer punto
        b (tuple): Coordenadas (x, y) del segundo punto
    
    Returns:
        int: Distancia Manhattan entre los puntos a y b
    
    Example:
        >>> heuristic((0, 0), (3, 4))
        7
        >>> heuristic((1, 1), (1, 1))
        0
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid_size, agent_pos, goal_pos, poisons):
    """
    Implementa el algoritmo A* para encontrar el camino óptimo en una cuadrícula.
    
    Busca el camino más corto desde la posición del agente hasta el objetivo,
    evitando venenos y respetando los límites del grid. Utiliza una heurística
    admisible (distancia Manhattan) para garantizar optimalidad.
    
    Algoritmo A*:
        1. Mantiene conjunto abierto (por explorar) y cerrado (explorados)
        2. Evalúa nodos usando f(n) = g(n) + h(n):
           - g(n): Costo real desde inicio hasta nodo n
           - h(n): Heurística desde nodo n hasta objetivo
        3. Expande el nodo con menor f(n) hasta encontrar objetivo
        4. Reconstruye camino desde objetivo hasta inicio
    
    Args:
        grid_size (int): Tamaño de la cuadrícula (grid_size x grid_size)
        agent_pos (tuple): Posición inicial del agente (x, y)
        goal_pos (tuple): Posición objetivo a alcanzar (x, y)
        poisons (list): Lista de posiciones de venenos [(x, y), ...]
    
    Returns:
        list: Secuencia de posiciones [(x, y), ...] del camino óptimo,
              excluyendo posición inicial. None si no existe camino.
    
    Example:
        >>> a_star_search(5, (0, 0), (4, 4), [(2, 2)])
        [(1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    
    Note:
        - Movimiento limitado a 4 direcciones (arriba, abajo, izquierda, derecha)
        - Venenos son obstáculos impasables
        - Retorna None si el objetivo es inalcanzable
    """
    # Convertir posiciones a tuplas para consistencia
    start = tuple(agent_pos)
    goal = tuple(goal_pos)
    
    # Inicializar estructuras de datos del algoritmo A*
    close_set = set()           # Nodos ya explorados
    came_from = {}              # Mapeo para reconstruir camino
    g_score = {start: 0}        # Costo real desde inicio
    f_score = {start: heuristic(start, goal)}  # Costo estimado total
    
    # Heap para mantener nodos ordenados por f_score
    open_heap = [(f_score[start], start)]

    # Bucle principal del algoritmo A*
    while open_heap:
        # Extraer nodo con menor f_score
        current = heapq.heappop(open_heap)[1]

        # ¿Hemos llegado al objetivo?
        if current == goal:
            # Reconstruir camino desde objetivo hasta inicio
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.insert(0, current)
            # Excluir posición inicial del camino retornado
            path.pop(0)
            return path

        # Marcar nodo actual como explorado
        close_set.add(current)
        
        # Explorar todos los vecinos (4 direcciones)
        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + i, current[1] + j)
            
            # Verificar límites del grid
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            
            # Verificar obstáculos: nodos explorados y venenos
            if neighbor in close_set or any(tuple(p) == neighbor for p in poisons):
                continue

            # Calcular nuevo costo para llegar al vecino
            tentative_g_score = g_score[current] + 1
            
            # ¿Es este un mejor camino al vecino?
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # Registrar mejor camino encontrado
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                # Agregar vecino a conjunto de exploración
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    
    # No se encontró camino al objetivo
    return None