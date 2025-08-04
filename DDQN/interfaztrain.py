# ddqn_agente_comefrutas.py
import pygame
import numpy as np
import os
import time
from agent import Agent
from environment import GridEnvironment

GRID_SIZE = 5
CELL_SIZE = 120
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE

COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_CURSOR = (255, 255, 0)
COLOR_TEXTO = (230, 230, 230)

NUM_EPISODIOS_ENTRENAMIENTO = 3000
BATCH_SIZE = 128


def cargar_imagen(ruta, color_si_falla):
    try:
        img = pygame.image.load(ruta).convert_alpha()
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
        surf.fill(color_si_falla)
        return surf


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente Come-Frutas DDQN 🍓☠️")

    # Cargar imágenes
    img_fruta = cargar_imagen("fruta.png", (40, 200, 40))
    img_veneno = cargar_imagen("veneno.png", (200, 40, 40))
    img_pared = cargar_imagen("pared.png", (100, 100, 100))
    img_agente = cargar_imagen("agente.png", (40, 200, 40))

    entorno = GridEnvironment(size=GRID_SIZE)
    agente = Agent(state_shape=(3, GRID_SIZE, GRID_SIZE), action_size=4)

    cursor_pos = [0, 0]
    modo_juego = "SETUP"
    reloj = pygame.time.Clock()
    corriendo = True

    frutas = set()
    venenos = set()
    paredes = set()

    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_t:
                    if modo_juego != "TRAINING":
                        print("--- ENTRENANDO DDQN ---")
                        modo_juego = "TRAINING"
                        for episodio in range(NUM_EPISODIOS_ENTRENAMIENTO):
                            estado = entorno.reset(
                                agent_pos=(0, 0),
                                fruit_pos=list(frutas),
                                poison_pos=list(venenos),
                            )
                            terminado = False
                            total_reward = 0
                            while not terminado:
                                accion = agente.choose_action(estado, explore=True)
                                nuevo_estado, recompensa, terminado = entorno.step(
                                    accion
                                )
                                agente.remember(
                                    estado, accion, recompensa, nuevo_estado, terminado
                                )
                                agente.replay(BATCH_SIZE)
                                if agente.steps_done % agente.update_target_every == 0:
                                    agente.update_target_network()
                                estado = nuevo_estado
                                total_reward += recompensa
                            if (episodio + 1) % 100 == 0:
                                print(
                                    f"Ep {episodio+1}, Reward: {total_reward:.2f}, Epsilon: {agente.epsilon:.3f}"
                                )
                        print("--- ENTRENAMIENTO COMPLETO ---")
                        modo_juego = "PLAYING"

                elif evento.key == pygame.K_p:
                    print("--- MODO PLAYING ---")
                    entorno.reset(
                        agent_pos=(0, 0),
                        fruit_pos=list(frutas),
                        poison_pos=list(venenos),
                    )
                    modo_juego = "PLAYING"

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                if modo_juego == "SETUP":
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_SIZE - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_SIZE - 1, cursor_pos[0] + 1)

                    pos = tuple(cursor_pos[::-1])
                    if evento.key == pygame.K_f:
                        if pos in frutas:
                            frutas.remove(pos)
                        else:
                            frutas.add(pos)
                            venenos.discard(pos)
                            paredes.discard(pos)
                    elif evento.key == pygame.K_v:
                        if pos in venenos:
                            venenos.remove(pos)
                        else:
                            venenos.add(pos)
                            frutas.discard(pos)
                            paredes.discard(pos)
                    elif evento.key == pygame.K_w:
                        if pos in paredes:
                            paredes.remove(pos)
                        else:
                            paredes.add(pos)
                            frutas.discard(pos)
                            venenos.discard(pos)
                    elif evento.key == pygame.K_c:
                        frutas.clear()
                        venenos.clear()
                        paredes.clear()

        if modo_juego == "PLAYING":
            estado = entorno.get_state()
            accion = agente.choose_action(estado, explore=False)
            _, _, terminado = entorno.step(accion)
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
            time.sleep(0.1)

        # --- Dibujo ---
        pantalla.fill(COLOR_FONDO)
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        for pared in paredes:
            pantalla.blit(img_pared, (pared[1] * CELL_SIZE, pared[0] * CELL_SIZE))
        for fruta in frutas:
            pantalla.blit(img_fruta, (fruta[1] * CELL_SIZE, fruta[0] * CELL_SIZE))
        for veneno in venenos:
            pantalla.blit(img_veneno, (veneno[1] * CELL_SIZE, veneno[0] * CELL_SIZE))

        if modo_juego != "SETUP":
            pos = entorno.agent_pos
            pantalla.blit(img_agente, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE))

        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        font = pygame.font.Font(None, 24)
        pantalla.blit(
            font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO),
            (10, SCREEN_HEIGHT + 5),
        )
        pantalla.blit(
            font.render(
                "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar",
                True,
                COLOR_TEXTO,
            ),
            (10, SCREEN_HEIGHT + 30),
        )
        pantalla.blit(
            font.render("T=Entrenar, P=Jugar, S=Setup", True, COLOR_TEXTO),
            (10, SCREEN_HEIGHT + 55),
        )

        pygame.display.flip()
        reloj.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
