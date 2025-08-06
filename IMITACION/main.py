# main.py
"""
Demostración simple del agente entrenado por aprendizaje por imitación.

Este módulo proporciona una interfaz minimalista para configurar escenarios
y observar el comportamiento del agente entrenado. Utiliza formas geométricas
simples para representar elementos, enfocándose en la funcionalidad core
sin distracciones visuales complejas.

Características:
    - Configuración interactiva con mouse (clic izquierdo=fruta, clic derecho=veneno)
    - Demostración automática del agente entrenado
    - Representación visual simple con formas geométricas
    - Ciclo continuo configuración → demostración → reset

Constantes:
    GRID_SIZE: Tamaño del entorno (5x5)
    CELL_SIZE: Tamaño de cada celda en píxeles (100px)
    WIDTH, HEIGHT: Dimensiones de la ventana (500x500)
    COLOR_*: Esquema de colores para elementos visuales

Funciones:
    draw_elements: Renderizado de elementos con formas geométricas
    main: Bucle principal con modos configuración y demostración
"""
import pygame
import numpy as np
from environment import GridEnvironment
from agent import Agent

# Configuración del entorno y ventana
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

# Esquema de colores simple y claro
COLOR_GRID = (200, 200, 200)    # Gris claro para grid
COLOR_AGENT = (0, 0, 255)       # Azul para agente
COLOR_FRUIT = (0, 255, 0)       # Verde para frutas
COLOR_POISON = (255, 0, 0)      # Rojo para venenos

def draw_elements(win, agent_pos, fruits, poisons):
    """
    Renderiza todos los elementos del entorno usando formas geométricas simples.
    
    Dibuja el grid de navegación y representa cada elemento del entorno
    con formas distintivas: rectángulos para agente, círculos para frutas
    y cuadrados pequeños para venenos. Diseño minimalista para claridad.
    
    Representación visual:
        - Agente: Rectángulo azul de celda completa
        - Frutas: Círculos verdes centrados (1/3 del tamaño de celda)
        - Venenos: Cuadrados rojos con margen (80% del tamaño de celda)
        - Grid: Líneas grises para delimitación de celdas
    
    Args:
        win (pygame.Surface): Superficie donde renderizar
        agent_pos (np.ndarray): Posición del agente [fila, columna]
        fruits (list): Lista de posiciones de frutas [(fila, col), ...]
        poisons (list): Lista de posiciones de venenos [(fila, col), ...]
    
    Note:
        Convierte coordenadas (fila, columna) a píxeles (x, y) para Pygame.
        Agente en posición (-1, -1) no se dibuja (modo configuración).
    """
    # Limpiar pantalla con fondo negro
    win.fill((0,0,0))
    
    # Dibujar grid de navegación
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(win, COLOR_GRID, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(win, COLOR_GRID, (0, y), (WIDTH, y))
    # Dibujar agente (solo si posición válida)
    if agent_pos[0] >= 0 and agent_pos[1] >= 0:
        pygame.draw.rect(win, COLOR_AGENT, 
                        (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, 
                         CELL_SIZE, CELL_SIZE))
    
    # Dibujar frutas como círculos verdes
    for f in fruits:
        center_x = f[1] * CELL_SIZE + CELL_SIZE//2
        center_y = f[0] * CELL_SIZE + CELL_SIZE//2
        radius = CELL_SIZE//3
        pygame.draw.circle(win, COLOR_FRUIT, (center_x, center_y), radius)
    
    # Dibujar venenos como cuadrados rojos con margen
    for p in poisons:
        margin = 20  # Margen de 20px para distinguir de agente
        rect_x = p[1] * CELL_SIZE + margin
        rect_y = p[0] * CELL_SIZE + margin
        rect_size = CELL_SIZE - 2 * margin
        pygame.draw.rect(win, COLOR_POISON, (rect_x, rect_y, rect_size, rect_size))
    
    # Actualizar display para mostrar cambios
    pygame.display.update()

def main():
    """
    Función principal de la demostración simple del agente por imitación.
    
    Implementa un ciclo de dos modos: configuración interactiva donde el usuario
    coloca elementos con el mouse, y demostración automática donde el agente
    entrenado navega el escenario. Diseñado para evaluación rápida y directa
    del rendimiento del modelo.
    
    Flujo de la aplicación:
        1. Modo "setup": Usuario configura escenario con mouse
           - Clic izquierdo: Colocar fruta
           - Clic derecho: Colocar veneno
           - Espacio: Iniciar demostración
        
        2. Modo "run": Agente navega automáticamente
           - Inferencia con modelo entrenado
           - Movimiento automático cada 300ms
           - Terminación por victoria/derrota
           - Reset automático a configuración
    
    Controles:
        - Clic izquierdo: Agregar fruta en posición del mouse
        - Clic derecho: Agregar veneno en posición del mouse
        - Espacio: Iniciar demostración (solo si hay frutas)
        - Automático: Reset a configuración al terminar episodio
    
    Note:
        Requiere modelo entrenado "imitacion_model.pth" en directorio actual.
        El agente siempre inicia en posición (0,0) del grid.
    """
    # Inicializar Pygame y configurar ventana
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agente Come-Frutas (IA)")
    
    # Inicializar entorno y agente con modelo pre-entrenado
    env = GridEnvironment(size=GRID_SIZE)
    agent = Agent()
    agent.load_model("imitacion_model.pth")

    # Variables de estado de la aplicación
    fruits, poisons = [], []  # Listas de posiciones de elementos
    mode = "setup"           # Modo inicial: configuración
    run = True              # Control del bucle principal
    # Bucle principal de la aplicación
    while run:
        # Procesar eventos de entrada
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Lógica específica del modo configuración
            if mode == "setup":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Convertir posición del mouse a coordenadas de grid
                    pos = pygame.mouse.get_pos()
                    col, row = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
                    
                    # Clic izquierdo: Agregar fruta (si no existe)
                    if event.button == 1 and (row, col) not in fruits:
                        fruits.append((row, col))
                    # Clic derecho: Agregar veneno (si no existe)
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))
                
                # Espacio: Iniciar demostración si hay frutas configuradas
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if fruits:  # Solo si hay al menos una fruta
                        mode = "run"
                        # Inicializar entorno con configuración actual
                        env.reset(agent_pos=(0,0), fruit_pos=fruits, poison_pos=poisons)

        # Renderizado según el modo actual
        if mode == "setup":
            # Modo configuración: mostrar elementos sin agente
            draw_elements(win, np.array([-1,-1]), fruits, poisons)
            
        elif mode == "run":
            # Modo demostración: agente automático
            
            # Obtener estado actual y generar acción
            state = env.get_state()
            action = agent.choose_action(state)
            
            # Ejecutar acción y verificar terminación
            _, _, done = env.step(action)
            
            # Renderizar estado actualizado
            draw_elements(win, env.agent_pos, env.fruit_pos, env.poison_pos)
            
            # Procesar terminación del episodio
            if done:
                print("¡Simulación terminada!")
                pygame.time.delay(2000)  # Pausa para observar resultado final
                # Reset automático a modo configuración
                fruits, poisons = [], []
                mode = "setup"
            
            # Controlar velocidad de demostración
            pygame.time.delay(300)  # 300ms entre acciones para visibilidad

    # Limpiar recursos al salir
    pygame.quit()


if __name__ == "__main__":
    main()