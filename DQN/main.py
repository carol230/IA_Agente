# main.py
import pygame
import numpy as np
from environment import GridEnvironment
from agent import Agent

# --- Configuración de Pygame ---
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agente Come-Frutas")
pygame.font.init()

# --- Cargar Agente Entrenado ---
env = GridEnvironment(size=GRID_SIZE)
action_size = 4
state_shape = (3, GRID_SIZE, GRID_SIZE)  # <-- CAMBIO: Usar la forma del estado
agent = Agent(state_shape, action_size)    # <-- CAMBIO: Pasar la tupla de forma
agent.load("dqn_model.pth")

# --- Colores y Assets (puedes usar imágenes) ---
COLOR_GRID = (200, 200, 200)
COLOR_AGENT = (0, 0, 255)
COLOR_FRUIT = (0, 255, 0)
COLOR_POISON = (255, 0, 0)

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(WIN, COLOR_GRID, (0, y), (WIDTH, y))

def draw_elements(agent_pos, fruits, poisons):
    # Dibujar agente
    pygame.draw.rect(WIN, COLOR_AGENT, (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Dibujar frutas
    for f in fruits:
        pygame.draw.circle(WIN, COLOR_FRUIT, (f[1] * CELL_SIZE + CELL_SIZE//2, f[0] * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)
    # Dibujar venenos
    for p in poisons:
        pygame.draw.rect(WIN, COLOR_POISON, (p[1] * CELL_SIZE + 20, p[0] * CELL_SIZE + 20, CELL_SIZE - 40, CELL_SIZE - 40))

def main():
    fruits = []
    poisons = []
    mode = "setup" # "setup" o "run"

    clock = pygame.time.Clock()
    run = True
    while run:
        WIN.fill((0,0,0))
        draw_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if mode == "setup":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    col = pos[0] // CELL_SIZE
                    row = pos[1] // CELL_SIZE
                    
                    # Click izquierdo para fruta
                    if event.button == 1 and (row, col) not in fruits:
                        fruits.append((row, col))
                    # Click derecho para veneno
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        mode = "run"
                        # Iniciar el juego con la configuración del usuario
                        state = env.reset(agent_pos=(0,0), fruit_pos=fruits, poison_pos=poisons)
                        print("Iniciando simulación...")

        if mode == "setup":
            # Dibujar elementos colocados por el usuario
            draw_elements(np.array([-1,-1]), fruits, poisons) # Agente fuera de pantalla
        
        elif mode == "run":
            # --- El agente toma el control ---
            state = env.get_state()
            action = agent.choose_action(state, explore=False)
            next_state, reward, done = env.step(action)
            
            draw_elements(env.agent_pos, env.fruit_pos, env.poison_pos)

            if done:
                print("¡Simulación terminada!")
                # Reiniciar para que otro visitante juegue
                fruits = []
                poisons = []
                mode = "setup"
                pygame.time.delay(2000)

            pygame.time.delay(300) # Pausa para que se vea el movimiento

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()