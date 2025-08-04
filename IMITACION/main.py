# main.py
import pygame
import numpy as np
from environment import GridEnvironment
from agent import Agent

GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

COLOR_GRID = (200, 200, 200)
COLOR_AGENT = (0, 0, 255)
COLOR_FRUIT = (0, 255, 0)
COLOR_POISON = (255, 0, 0)

def draw_elements(win, agent_pos, fruits, poisons):
    win.fill((0,0,0))
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(win, COLOR_GRID, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(win, COLOR_GRID, (0, y), (WIDTH, y))
    
    pygame.draw.rect(win, COLOR_AGENT, (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for f in fruits:
        pygame.draw.circle(win, COLOR_FRUIT, (f[1] * CELL_SIZE + CELL_SIZE//2, f[0] * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)
    for p in poisons:
        pygame.draw.rect(win, COLOR_POISON, (p[1] * CELL_SIZE + 20, p[0] * CELL_SIZE + 20, CELL_SIZE - 40, CELL_SIZE - 40))
    pygame.display.update()

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agente Come-Frutas (IA)")
    
    env = GridEnvironment(size=GRID_SIZE)
    agent = Agent()
    agent.load_model("imitacion_model.pth")

    fruits, poisons = [], []
    mode = "setup"
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if mode == "setup":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    col, row = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
                    if event.button == 1 and (row, col) not in fruits:
                        fruits.append((row, col))
                    elif event.button == 3 and (row, col) not in poisons:
                        poisons.append((row, col))
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if fruits:
                        mode = "run"
                        env.reset(agent_pos=(0,0), fruit_pos=fruits, poison_pos=poisons)

        if mode == "setup":
            draw_elements(win, np.array([-1,-1]), fruits, poisons)
        elif mode == "run":
            state = env.get_state()
            action = agent.choose_action(state)
            _, _, done = env.step(action)
            draw_elements(win, env.agent_pos, env.fruit_pos, env.poison_pos)
            
            if done:
                print("¡Simulación terminada!")
                pygame.time.delay(2000)
                fruits, poisons = [], []
                mode = "setup"
            
            pygame.time.delay(300)

    pygame.quit()

if __name__ == "__main__":
    main()