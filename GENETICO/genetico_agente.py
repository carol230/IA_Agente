import pygame
import numpy as np
import os
import time
from agent_ga import Agent

GRID_WIDTH = 5
GRID_HEIGHT = 5
CELL_SIZE = 120
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_CURSOR = (255, 255, 0)
COLOR_TEXTO = (230, 230, 230)


class EntornoGrid:
    def __init__(self):
        self.size = GRID_WIDTH
        self.agent_pos = (0, 0)
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset_a_configuracion_inicial(self):
        self.agent_pos = (0, 0)
        return self.get_state()

    def limpiar_entorno(self):
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        fila, col = self.agent_pos
        if accion == 0:
            fila -= 1
        elif accion == 1:
            fila += 1
        elif accion == 2:
            col -= 1
        elif accion == 3:
            col += 1

        if (
            fila < 0
            or fila >= GRID_HEIGHT
            or col < 0
            or col >= GRID_WIDTH
            or (fila, col) in self.paredes
        ):
            return self.get_state(), -0.1, False

        self.agent_pos = (fila, col)
        recompensa = -0.05
        terminado = False

        if self.agent_pos in self.venenos:
            recompensa = -10.0
            self.agent_pos = (0, 0)
        elif self.agent_pos in self.frutas:
            recompensa = 1.0
            self.frutas.remove(self.agent_pos)
            if not self.frutas:
                recompensa += 10.0
                terminado = True
                self.agent_pos = (0, 0)

        return self.get_state(), recompensa, terminado

    def get_state(self):
        estado = np.zeros((3, self.size, self.size), dtype=np.float32)
        estado[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        for fruta in self.frutas:
            estado[1, fruta[0], fruta[1]] = 1.0
        for veneno in self.venenos:
            estado[2, veneno[0], veneno[1]] = 1.0
        return estado

    def dibujar(
        self,
        pantalla,
        modo_juego,
        cursor_pos,
        img_fruta,
        img_veneno,
        img_pared,
        img_agente,
    ):
        pantalla.fill(COLOR_FONDO)
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0] * CELL_SIZE, pared[1] * CELL_SIZE))
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0] * CELL_SIZE, fruta[1] * CELL_SIZE))
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0] * CELL_SIZE, veneno[1] * CELL_SIZE))

        pantalla.blit(
            img_agente, (self.agent_pos[0] * CELL_SIZE, self.agent_pos[1] * CELL_SIZE)
        )

        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        font = pygame.font.Font(None, 24)
        texto_modo = font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO)
        controles1 = font.render(
            "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar", True, COLOR_TEXTO
        )
        controles2 = font.render("P=Jugar, S=Setup", True, COLOR_TEXTO)
        pantalla.blit(texto_modo, (10, SCREEN_HEIGHT + 5))
        pantalla.blit(controles1, (10, SCREEN_HEIGHT + 30))
        pantalla.blit(controles2, (10, SCREEN_HEIGHT + 55))


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente Genético - Come Frutas 🍓")

    def cargar_img(nombre, color_fallback):
        try:
            ruta = os.path.join(os.path.dirname(__file__), nombre)
            img = pygame.image.load(ruta).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except:
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.fill(color_fallback)
            return surf

    img_fruta = cargar_img("../fruta.png", (0, 255, 0))
    img_veneno = cargar_img("../veneno.png", (255, 0, 0))
    img_pared = cargar_img("../pared.png", (100, 100, 100))
    img_agente = cargar_img("../agente.png", (0, 0, 255))

    entorno = EntornoGrid()
    agente = Agent()
    agente.load_genes("GENETICO/best_agent_genes.pth")

    cursor_pos = [0, 0]
    modo_juego = "SETUP"
    reloj = pygame.time.Clock()
    corriendo = True

    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_p:
                    print("--- MODO JUEGO ---")
                    entorno.reset_a_configuracion_inicial()
                    modo_juego = "PLAYING"
                    time.sleep(0.5)

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                if modo_juego == "SETUP":
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_HEIGHT - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_WIDTH - 1, cursor_pos[0] + 1)

                    pos = tuple(cursor_pos)
                    if evento.key == pygame.K_f:
                        if pos in entorno.frutas:
                            entorno.frutas.remove(pos)
                        else:
                            entorno.frutas.add(pos)
                            entorno.venenos.discard(pos)
                            entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_v:
                        if pos in entorno.venenos:
                            entorno.venenos.remove(pos)
                        else:
                            entorno.venenos.add(pos)
                            entorno.frutas.discard(pos)
                            entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_w:
                        if pos in entorno.paredes:
                            entorno.paredes.remove(pos)
                        else:
                            entorno.paredes.add(pos)
                            entorno.frutas.discard(pos)
                            entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_c:
                        print("--- LIMPIANDO ENTORNO ---")
                        entorno.limpiar_entorno()

        if modo_juego == "PLAYING":
            estado = entorno.get_state()
            accion = agente.choose_action(estado)
            _, _, terminado = entorno.step(accion)
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
            time.sleep(0.1)

        pantalla_con_info = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
        pantalla_con_info.fill(COLOR_FONDO)
        entorno.dibujar(
            pantalla_con_info,
            modo_juego,
            tuple(cursor_pos),
            img_fruta,
            img_veneno,
            img_pared,
            img_agente,
        )
        pantalla.blit(pantalla_con_info, (0, 0))
        pygame.display.flip()
        reloj.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
