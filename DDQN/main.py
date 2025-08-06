# main.py
"""
Demostración interactiva del agente DDQN entrenado.

Este módulo proporciona una interfaz de demostración simple donde los usuarios pueden:
- Configurar entornos personalizados mediante clics del mouse
- Observar el comportamiento del agente DDQN entrenado en tiempo real
- Experimentar con diferentes configuraciones de frutas y venenos

El sistema está diseñado como una demostración pública o para validación
del rendimiento del agente en escenarios definidos por el usuario.

Características principales:
- Interfaz minimalista y fácil de usar
- Carga automática del modelo entrenado
- Visualización en tiempo real del agente
- Reinicio automático para múltiples demostraciones

Flujo de trabajo:
1. Modo SETUP: El usuario configura frutas y venenos con clics
2. Modo RUN: El agente ejecuta la solución automáticamente
3. Reinicio automático al completar la demostración
"""

import pygame
import numpy as np
from environment import GridEnvironment
from agent import Agent

# --- CONFIGURACIÓN DE LA INTERFAZ GRÁFICA ---
"""Parámetros de visualización y configuración de la ventana."""
GRID_SIZE = 5           # Tamaño de la grilla (5x5)
CELL_SIZE = 100         # Tamaño de cada celda en píxeles
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE  # Dimensiones de ventana
WIN = pygame.display.set_mode((WIDTH, HEIGHT))               # Ventana principal
pygame.display.set_caption("Agente Come-Frutas")            # Título de la ventana
pygame.font.init()                                           # Inicializar sistema de fuentes

# --- INICIALIZACIÓN DEL AGENTE ENTRENADO ---
"""
Configuración y carga del agente DDQN preentrenado.

Este bloque inicializa los componentes necesarios para ejecutar
el agente entrenado en modo demostración.
"""
env = GridEnvironment(size=GRID_SIZE)                    # Entorno de simulación
action_size = 4                                          # Número de acciones posibles (4 direcciones)
state_shape = (3, GRID_SIZE, GRID_SIZE)                 # Forma del estado: 3 canales x 5x5
agent = Agent(state_shape, action_size)                 # Crear instancia del agente
agent.load("dqn_model.pth")                            # Cargar modelo preentrenado

# --- ESQUEMA DE COLORES PARA VISUALIZACIÓN ---
"""Colores RGB para los diferentes elementos del juego."""
COLOR_GRID = (200, 200, 200)    # Gris claro para las líneas de la grilla
COLOR_AGENT = (0, 0, 255)       # Azul para el agente
COLOR_FRUIT = (0, 255, 0)       # Verde para las frutas (objetivos)
COLOR_POISON = (255, 0, 0)      # Rojo para los venenos (peligros)


def draw_grid():
    """
    Dibuja las líneas de la grilla en la ventana.
    
    Crea una cuadrícula visual que ayuda a los usuarios a identificar
    las posiciones disponibles para colocar elementos durante el modo setup.
    
    La grilla se dibuja con líneas verticales y horizontales espaciadas
    uniformemente según el tamaño de celda configurado.
    """
    # Dibujar líneas verticales
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (x, 0), (x, HEIGHT))
    
    # Dibujar líneas horizontales  
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (0, y), (WIDTH, y))


def draw_elements(agent_pos, fruits, poisons):
    """
    Renderiza todos los elementos del juego en la pantalla.
    
    Esta función se encarga de dibujar visualmente todos los componentes
    del entorno: el agente, las frutas y los venenos, usando formas
    geométricas distintivas para cada tipo de elemento.
    
    Args:
        agent_pos (np.array): Posición actual del agente [fila, columna]
        fruits (list): Lista de posiciones de frutas [(fila, col), ...]
        poisons (list): Lista de posiciones de venenos [(fila, col), ...]
    
    Representaciones visuales:
        - Agente: Rectángulo azul que ocupa toda la celda
        - Frutas: Círculos verdes centrados en las celdas
        - Venenos: Cuadrados rojos más pequeños dentro de las celdas
    """
    # Dibujar agente como rectángulo azul
    # Nota: Se intercambian coordenadas (agent_pos[1], agent_pos[0]) para
    # convertir de coordenadas de matriz (fila, columna) a pantalla (x, y)
    pygame.draw.rect(
        WIN,
        COLOR_AGENT,
        (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
    )
    
    # Dibujar frutas como círculos verdes
    for f in fruits:
        center_x = f[1] * CELL_SIZE + CELL_SIZE // 2  # Centro horizontal
        center_y = f[0] * CELL_SIZE + CELL_SIZE // 2  # Centro vertical
        radius = CELL_SIZE // 3                       # Radio del círculo
        pygame.draw.circle(WIN, COLOR_FRUIT, (center_x, center_y), radius)
    
    # Dibujar venenos como cuadrados rojos más pequeños
    for p in poisons:
        # Crear un margen de 20 píxeles alrededor del cuadrado
        x = p[1] * CELL_SIZE + 20
        y = p[0] * CELL_SIZE + 20
        size = CELL_SIZE - 40  # Tamaño reducido del cuadrado
        pygame.draw.rect(WIN, COLOR_POISON, (x, y, size, size))


def main():
    """
    Función principal que ejecuta la demostración interactiva del agente DDQN.
    
    Esta función implementa un sistema de dos modos que permite a los usuarios
    configurar entornos personalizados y observar el comportamiento del agente.
    
    Modos de operación:
    
    1. **MODO SETUP** (Configuración interactiva):
       - Permite al usuario colocar elementos usando el mouse
       - Clic izquierdo: Añadir frutas (objetivos)
       - Clic derecho: Añadir venenos (obstáculos peligrosos)
       - Barra espaciadora: Iniciar simulación
    
    2. **MODO RUN** (Demostración del agente):
       - El agente DDQN toma control total
       - Ejecuta acciones basadas en el modelo entrenado
       - Visualización en tiempo real del comportamiento
       - Finalización automática y reinicio
    
    El sistema está diseñado para demostraciones públicas, permitiendo
    múltiples usuarios configurar y probar diferentes escenarios.
    """
    # Variables de estado del sistema
    fruits = []           # Lista de posiciones de frutas configuradas por el usuario
    poisons = []          # Lista de posiciones de venenos configuradas por el usuario
    mode = "setup"        # Modo inicial: "setup" para configuración, "run" para ejecución

    # Configuración del bucle principal
    clock = pygame.time.Clock()  # Control de framerate
    run = True                   # Flag principal del bucle

    # Bucle principal de la demostración
    while run:
        # Limpiar pantalla con fondo negro
        WIN.fill((0, 0, 0))
        
        # Dibujar grilla de referencia
        draw_grid()

        # Procesar eventos del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # --- LÓGICA DEL MODO SETUP ---
            if mode == "setup":
                # Manejo de clics del mouse para colocar elementos
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    col = pos[0] // CELL_SIZE  # Convertir coordenada x a columna
                    row = pos[1] // CELL_SIZE  # Convertir coordenada y a fila

                    # Clic izquierdo: Añadir fruta (si no existe ya)
                    if event.button == 1 and (row, col) not in fruits:
                        fruits.append((row, col))
                        
                    # Clic derecho: Añadir veneno (si no existe ya)
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))

                # Tecla espaciadora: Iniciar simulación
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        mode = "run"
                        # Configurar el entorno con los elementos del usuario
                        state = env.reset(
                            agent_pos=(0, 0),      # Agente siempre inicia en (0,0)
                            fruit_pos=fruits,      # Frutas configuradas por el usuario
                            poison_pos=poisons     # Venenos configurados por el usuario
                        )
                        print("Iniciando simulación...")

        # --- RENDERIZADO SEGÚN EL MODO ACTUAL ---
        if mode == "setup":
            # Modo configuración: Mostrar elementos colocados por el usuario
            # El agente se dibuja fuera de pantalla (posición inválida) para ocultarlo
            draw_elements(
                np.array([-1, -1]),  # Posición fuera de pantalla para el agente
                fruits,              # Frutas configuradas por el usuario
                poisons              # Venenos configurados por el usuario
            )

        elif mode == "run":
            # --- LÓGICA DEL AGENTE AUTÓNOMO ---
            # Obtener estado actual del entorno
            state = env.get_state()
            
            # El agente elige la mejor acción sin exploración
            # explore=False asegura que use solo la política aprendida
            action = agent.choose_action(state, explore=False)
            
            # Ejecutar la acción en el entorno
            next_state, reward, done = env.step(action)

            # Renderizar estado actual con posiciones reales del entorno
            draw_elements(
                env.agent_pos,    # Posición actual del agente
                env.fruit_pos,    # Frutas restantes en el entorno
                env.poison_pos    # Venenos en el entorno
            )

            # Verificar si el episodio terminó
            if done:
                print("¡Simulación terminada!")
                # Reiniciar sistema para nueva demostración
                fruits = []        # Limpiar frutas configuradas
                poisons = []       # Limpiar venenos configurados
                mode = "setup"     # Volver al modo configuración
                pygame.time.delay(2000)  # Pausa de 2 segundos antes del reinicio

            # Control de velocidad de visualización
            pygame.time.delay(300)  # Pausa de 300ms para observar movimientos

        # Actualizar pantalla para mostrar cambios
        pygame.display.update()

    # Limpiar recursos al salir
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa de demostración.
    
    Ejecuta la función main() cuando el archivo se ejecuta directamente.
    Este patrón permite importar funciones de este módulo en otros scripts
    sin ejecutar automáticamente la demostración.
    
    Uso típico:
        python main.py  # Ejecuta la demostración interactiva
        
    La demostración está diseñada para:
    - Presentaciones públicas del proyecto
    - Validación rápida del comportamiento del agente
    - Experimentación interactiva con diferentes configuraciones
    - Evaluación cualitativa del rendimiento del modelo
    """
    main()
