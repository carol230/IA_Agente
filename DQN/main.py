# main.py
"""
Interfaz gráfica interactiva para visualizar un agente DQN entrenado.

Este módulo implementa una aplicación Pygame que permite:
1. Configurar un escenario colocando frutas y venenos manualmente
2. Observar cómo el agente DQN entrenado resuelve el escenario
3. Reiniciar para probar diferentes configuraciones

La aplicación tiene dos modos:
- Modo Setup: El usuario coloca elementos en la cuadrícula
- Modo Run: El agente toma control y ejecuta su política aprendida

Controles:
- Click izquierdo: Colocar fruta
- Click derecho: Colocar veneno  
- Espacio: Iniciar simulación del agente

"""

import pygame
import numpy as np
from environment import GridEnvironment
from agent import Agent

# CONFIGURACIÓN DE PYGAME Y CONSTANTES DEL JUEGO
"""
Configuración visual y dimensiones de la aplicación.
"""
GRID_SIZE = 5        # Tamaño de la cuadrícula (5x5)
CELL_SIZE = 100      # Tamaño de cada celda en píxeles
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE  # Ventana de 500x500 píxeles
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agente Come-Frutas")
pygame.font.init()   # Inicializar fuentes para texto si se necesita

# INICIALIZACIÓN DEL AGENTE DQN ENTRENADO
"""
Carga el agente DQN previamente entrenado desde archivo.
El agente utilizará su política aprendida para navegar por el entorno.
"""
env = GridEnvironment(size=GRID_SIZE)
action_size = 4  # 4 acciones posibles: arriba, abajo, izquierda, derecha
state_shape = (3, GRID_SIZE, GRID_SIZE)  # Forma del estado: 3 canales x 5x5 grid
agent = Agent(state_shape, action_size)  # Crear instancia del agente
agent.load("dqn_model.pth")              # Cargar pesos del modelo entrenado

# DEFINICIÓN DE COLORES
"""
Paleta de colores para los elementos visuales del juego.
Utiliza sistema RGB (Red, Green, Blue) con valores 0-255.
"""
COLOR_GRID = (200, 200, 200)   # Gris claro para las líneas de la cuadrícula
COLOR_AGENT = (0, 0, 255)      # Azul para el agente
COLOR_FRUIT = (0, 255, 0)      # Verde para las frutas
COLOR_POISON = (255, 0, 0)     # Rojo para los venenos

def draw_grid():
    """
    Dibuja las líneas de la cuadrícula en la ventana.
    
    Crea una cuadrícula visual de 5x5 dibujando líneas verticales y horizontales
    separadas por CELL_SIZE píxeles. Esto ayuda a visualizar las celdas donde
    se pueden colocar elementos y donde se mueve el agente.
    """
    # Líneas verticales
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (x, 0), (x, HEIGHT))
    # Líneas horizontales  
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (0, y), (WIDTH, y))

def draw_elements(agent_pos, fruits, poisons):
    """
    Dibuja todos los elementos del juego en sus posiciones actuales.
    
    Renderiza visualmente:
    - Agente: Como un cuadrado azul que ocupa toda la celda
    - Frutas: Como círculos verdes centrados en sus celdas
    - Venenos: Como cuadrados rojos más pequeños centrados en sus celdas
    
    Args:
        agent_pos (np.array): Posición del agente [fila, columna]
        fruits (list): Lista de posiciones de frutas [(fila, col), ...]
        poisons (list): Lista de posiciones de venenos [(fila, col), ...]
    
    Note:
        Las coordenadas se invierten para Pygame: agent_pos[1] es X, agent_pos[0] es Y
    """
    # Dibujar agente como cuadrado azul completo
    if agent_pos[0] >= 0:  # Solo dibujar si el agente está en el tablero
        pygame.draw.rect(WIN, COLOR_AGENT, 
                        (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, 
                         CELL_SIZE, CELL_SIZE))
    
    # Dibujar frutas como círculos verdes
    for f in fruits:
        center_x = f[1] * CELL_SIZE + CELL_SIZE // 2
        center_y = f[0] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        pygame.draw.circle(WIN, COLOR_FRUIT, (center_x, center_y), radius)
    
    # Dibujar venenos como cuadrados rojos más pequeños
    for p in poisons:
        margin = 20  # Margen para hacer el cuadrado más pequeño
        pygame.draw.rect(WIN, COLOR_POISON, 
                        (p[1] * CELL_SIZE + margin, p[0] * CELL_SIZE + margin, 
                         CELL_SIZE - 2*margin, CELL_SIZE - 2*margin))

def main():
    """
    Función principal que maneja el bucle de la aplicación.
    
    Implementa una máquina de estados con dos modos:
    
    MODO SETUP:
    - Permite al usuario colocar frutas y venenos con clics del mouse
    - Click izquierdo: Colocar fruta
    - Click derecho: Colocar veneno
    - Presionar ESPACIO: Iniciar simulación
    
    MODO RUN:
    - El agente DQN toma control del juego
    - Ejecuta acciones basadas en su política aprendida
    - Visualiza el comportamiento del agente en tiempo real
    - Se reinicia automáticamente al terminar
    
    La aplicación se ejecuta hasta que el usuario cierre la ventana.
    """
    # Variables de estado del juego
    fruits = []      # Lista de posiciones de frutas colocadas por el usuario
    poisons = []     # Lista de posiciones de venenos colocadas por el usuario  
    mode = "setup"   # Modo actual: "setup" (configuración) o "run" (simulación)

    # Configuración del bucle principal
    clock = pygame.time.Clock()  # Para controlar FPS
    run = True                   # Flag de control del bucle principal
    # BUCLE PRINCIPAL DE LA APLICACIÓN
    while run:
        # Limpiar pantalla con fondo negro
        WIN.fill((0, 0, 0))
        # Dibujar cuadrícula base
        draw_grid()

        # MANEJO DE EVENTOS DE USUARIO
        for event in pygame.event.get():
            # Evento de cierre de ventana
            if event.type == pygame.QUIT:
                run = False

            # EVENTOS EN MODO SETUP (Configuración manual)
            if mode == "setup":
                # Manejo de clics del mouse para colocar elementos
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    # Convertir coordenadas de píxeles a coordenadas de cuadrícula
                    col = pos[0] // CELL_SIZE
                    row = pos[1] // CELL_SIZE
                    
                    # Click izquierdo (botón 1): Colocar fruta
                    if event.button == 1 and (row, col) not in fruits:
                        fruits.append((row, col))
                        print(f"Fruta colocada en ({row}, {col})")
                    
                    # Click derecho (botón 3): Colocar veneno  
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))
                        print(f"Veneno colocado en ({row}, {col})")

                # Manejo de teclas para cambiar de modo
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        mode = "run"
                        # Inicializar el entorno con la configuración del usuario
                        state = env.reset(agent_pos=(0, 0), fruit_pos=fruits, poison_pos=poisons)
                        print("=== INICIANDO SIMULACIÓN DEL AGENTE ===")
                        print(f"Frutas: {len(fruits)}, Venenos: {len(poisons)}")

        # RENDERIZADO SEGÚN EL MODO ACTUAL
        
        if mode == "setup":
            # MODO CONFIGURACIÓN: Mostrar elementos colocados por el usuario
            # Usar posición (-1,-1) para que el agente no aparezca en pantalla
            draw_elements(np.array([-1, -1]), fruits, poisons)
        
        elif mode == "run":
            # MODO SIMULACIÓN: El agente DQN ejecuta su política
            
            # Obtener estado actual del entorno
            state = env.get_state()
            
            # El agente decide la acción usando su política entrenada
            # explore=False significa que usa solo explotación, no exploración
            action = agent.choose_action(state, explore=False)
            
            # Ejecutar la acción en el entorno
            next_state, reward, done = env.step(action)
            
            # Renderizar estado actual del juego
            draw_elements(env.agent_pos, env.fruit_pos, env.poison_pos)

            # Verificar si el episodio terminó
            if done:
                if not env.fruit_pos:  # Victoria: todas las frutas recogidas
                    print("🎉 ¡ÉXITO! El agente recogió todas las frutas")
                else:  # Derrota: tocó veneno
                    print("💀 DERROTA: El agente tocó veneno")
                
                print("=== SIMULACIÓN TERMINADA ===")
                
                # Reiniciar para permitir nueva configuración
                fruits = []
                poisons = []
                mode = "setup"
                
                # Pausa dramática antes de reiniciar
                pygame.time.delay(2000)

            # Pausa entre movimientos para visualización clara
            pygame.time.delay(300)

        # Actualizar pantalla con todos los cambios
        pygame.display.update()

    # Limpieza al cerrar la aplicación
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa.
    
    Ejecuta la función main() solo si este archivo se ejecuta directamente
    (no si se importa como módulo).
    """
    print("=== AGENTE DQN COME-FRUTAS ===")
    print("CONTROLES:")
    print("• Click izquierdo: Colocar fruta")
    print("• Click derecho: Colocar veneno")
    print("• ESPACIO: Iniciar simulación")
    print("• Cerrar ventana: Salir")
    print("\n¡Configura un escenario y observa al agente!")
    
    main()