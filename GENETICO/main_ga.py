# main_ga.py
"""
Demostrador simplificado para agentes entrenados con algoritmos genéticos.

Este módulo implementa una interfaz gráfica básica que permite configurar
escenarios y observar el comportamiento de un agente genético entrenado.
Es una versión adaptada del demostrador DQN para algoritmos genéticos.

Características principales:
- Configuración manual con mouse (click izquierdo=fruta, derecho=veneno)
- Demostración automática del agente evolucionado
- Interfaz simple y directa
- Comportamiento determinístico del agente

Diferencias con DQN:
- Carga genes en lugar de modelo entrenado
- Sin parámetro explore en choose_action
- Agente completamente determinístico

Controles:
- Click izquierdo: Colocar fruta
- Click derecho: Colocar veneno
- Espacio: Iniciar simulación

Autor: [Tu nombre]
Fecha: Agosto 2025
"""

import pygame
from environment import GridEnvironment
from agent_ga import Agent 
import numpy as np  # Importa numpy para manejo de arrays

# CONFIGURACIÓN DE PYGAME Y VENTANA
"""
Parámetros básicos para la interfaz gráfica, idénticos al demostrador DQN
para mantener consistencia visual entre diferentes enfoques de IA.
"""
GRID_SIZE = 5        # Tamaño de la cuadrícula (5x5)
CELL_SIZE = 100      # Tamaño de cada celda en píxeles
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE  # Ventana 500x500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agente Come-Frutas")
pygame.font.init()

# INICIALIZACIÓN DEL AGENTE GENÉTICO
"""
Configuración del entorno y carga del agente con genes evolucionados.
A diferencia del DQN, aquí cargamos 'genes' (pesos) en lugar de un modelo entrenado.
"""
env = GridEnvironment(size=GRID_SIZE)
agent = Agent()
agent.load_genes("best_agent_genes.pth")  # Cargar los genes del mejor agente evolucionado

# PALETA DE COLORES PARA ELEMENTOS VISUALES
"""
Esquema de colores consistente con otros demostradores del proyecto
para facilitar la comparación entre diferentes enfoques de IA.
"""
COLOR_GRID = (200, 200, 200)   # Gris claro para líneas de cuadrícula
COLOR_AGENT = (0, 0, 255)      # Azul para el agente
COLOR_FRUIT = (0, 255, 0)      # Verde para frutas
COLOR_POISON = (255, 0, 0)     # Rojo para venenos

def draw_grid():
    """
    Dibuja las líneas de la cuadrícula de referencia.
    
    Crea una cuadrícula visual de 5x5 para ayudar al usuario a visualizar
    las posiciones disponibles para colocar elementos. Idéntica implementación
    a otros demostradores para consistencia visual.
    """
    # Líneas verticales
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (x, 0), (x, HEIGHT))
    # Líneas horizontales
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (0, y), (WIDTH, y))

def draw_elements(agent_pos, fruits, poisons):
    """
    Renderiza todos los elementos del juego en sus posiciones actuales.
    
    Dibuja el agente, frutas y venenos usando formas geométricas simples.
    Mantiene el mismo estilo visual que el demostrador DQN para facilitar
    la comparación entre enfoques.
    
    Args:
        agent_pos (np.array): Posición del agente [fila, columna]
        fruits (list): Lista de posiciones de frutas [(fila, col), ...]
        poisons (list): Lista de posiciones de venenos [(fila, col), ...]
    
    Note:
        Las coordenadas se invierten para Pygame: columna=X, fila=Y
    """
    # Dibujar agente como rectángulo azul completo
    if agent_pos[0] >= 0:  # Solo dibujar si está en el tablero
        pygame.draw.rect(WIN, COLOR_AGENT, 
                        (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, 
                         CELL_SIZE, CELL_SIZE))
    
    # Dibujar frutas como círculos verdes
    for f in fruits:
        center_x = f[1] * CELL_SIZE + CELL_SIZE // 2
        center_y = f[0] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        pygame.draw.circle(WIN, COLOR_FRUIT, (center_x, center_y), radius)
    
    # Dibujar venenos como rectángulos rojos más pequeños
    for p in poisons:
        margin = 20  # Margen para hacer el rectángulo más pequeño
        pygame.draw.rect(WIN, COLOR_POISON, 
                        (p[1] * CELL_SIZE + margin, p[0] * CELL_SIZE + margin, 
                         CELL_SIZE - 2*margin, CELL_SIZE - 2*margin))

def main():
    """
    Función principal que ejecuta el demostrador de algoritmos genéticos.
    
    Implementa una aplicación de dos modos similar al demostrador DQN:
    - Modo Setup: Configuración manual del escenario
    - Modo Run: Demostración del agente genético evolucionado
    
    Diferencias clave con DQN:
    - choose_action sin parámetro explore (comportamiento 100% determinístico)
    - Carga genes en lugar de modelo de red neuronal
    - Sin exploración durante la demostración
    
    La aplicación permite probar el comportamiento del agente en diferentes
    configuraciones para evaluar la efectividad de la evolución genética.
    """
    # Variables de estado de la aplicación
    fruits = []      # Lista de posiciones de frutas configuradas por el usuario
    poisons = []     # Lista de posiciones de venenos configuradas por el usuario
    mode = "setup"   # Modo actual: "setup" (configuración) o "run" (demostración)

    # Configuración del bucle principal
    clock = pygame.time.Clock()  # Control de FPS
    run = True                   # Flag de control del bucle
    # BUCLE PRINCIPAL DE LA APLICACIÓN
    while run:
        # Limpiar pantalla y dibujar cuadrícula base
        WIN.fill((0, 0, 0))
        draw_grid()

        # MANEJO DE EVENTOS DEL USUARIO
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
                        print(f"🍎 Fruta colocada en ({row}, {col})")
                    
                    # Click derecho (botón 3): Colocar veneno
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))
                        print(f"☠️  Veneno colocado en ({row}, {col})")

                # Manejo de teclas para cambiar de modo
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        mode = "run"
                        # Inicializar entorno con configuración del usuario
                        state = env.reset(agent_pos=(0, 0), fruit_pos=fruits, poison_pos=poisons)
                        print("=== INICIANDO DEMOSTRACIÓN DEL AGENTE GENÉTICO ===")
                        print(f"Configuración: {len(fruits)} frutas, {len(poisons)} venenos")

        # RENDERIZADO SEGÚN EL MODO ACTUAL
        
        if mode == "setup":
            # MODO CONFIGURACIÓN: Mostrar elementos colocados por el usuario
            # Usar posición (-1,-1) para que el agente no aparezca en pantalla
            draw_elements(np.array([-1, -1]), fruits, poisons)
        
        elif mode == "run":
            # MODO DEMOSTRACIÓN: El agente genético ejecuta su comportamiento
            
            # Obtener estado actual del entorno
            state = env.get_state()
            
            # DIFERENCIA CLAVE CON DQN: Sin parámetro explore
            # El agente genético es 100% determinístico
            action = agent.choose_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done = env.step(action)
            
            # Renderizar estado actual
            draw_elements(env.agent_pos, env.fruit_pos, env.poison_pos)

            # Verificar condición de terminación
            if done:
                if not env.fruit_pos:  # Victoria: todas las frutas recogidas
                    print("🏆 ¡ÉXITO! El agente genético recogió todas las frutas")
                else:  # Otra condición de terminación
                    print("🔄 Simulación terminada")
                
                print("=== DEMOSTRACIÓN COMPLETADA ===")
                
                # Reiniciar para nueva configuración
                fruits = []
                poisons = []
                mode = "setup"
                
                # Pausa antes de reiniciar
                pygame.time.delay(2000)

            # Pausa para visualización clara del movimiento
            pygame.time.delay(300)

        # Actualizar pantalla con todos los cambios
        pygame.display.update()

    # Limpieza al cerrar la aplicación
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa.
    
    Ejecuta la función main() solo si este archivo se ejecuta directamente.
    Incluye mensaje informativo sobre las diferencias con DQN.
    """
    print("=" * 60)
    print("🧬 DEMOSTRADOR DE AGENTE GENÉTICO (Versión Simplificada) 🧬")
    print("=" * 60)
    print("DIFERENCIAS CON DQN:")
    print("🔹 Comportamiento 100% determinístico (sin exploración)")
    print("🔹 Genes evolucionados vs. pesos entrenados por gradientes")
    print("🔹 Sin parámetros de aprendizaje durante la demostración")
    print()
    print("CONTROLES:")
    print("🖱️  Click izquierdo: Colocar fruta")
    print("🖱️  Click derecho: Colocar veneno")
    print("⌨️  ESPACIO: Iniciar demostración")
    print("❌ Cerrar ventana: Salir")
    print()
    print("🎯 Configura un escenario y observa la inteligencia evolucionada!")
    print("=" * 60)
    
    main()