import pygame
import os
import random
import string

# --- Configuración del entorno ---
GRID_WIDTH = 5
GRID_HEIGHT = 5
CELL_SIZE = 120
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

COLOR_FONDO = (25, 25, 25)
COLOR_LINEAS = (40, 40, 40)
COLOR_CURSOR = (255, 255, 0)
COLOR_TEXTO = (230, 230, 230)

# Se permiten solo letras y números (evitando teclas especiales)
TECLAS_VALIDAS = [getattr(pygame, f"K_{c}") for c in string.ascii_lowercase + string.digits]

class EntornoHumano:
    def __init__(self):
        self.agente_pos = (0, 0)
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset(self):
        self.agente_pos = (0, 0)

    def limpiar(self):
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        x, y = self.agente_pos
        if accion == 0: y -= 1
        elif accion == 1: y += 1
        elif accion == 2: x -= 1
        elif accion == 3: x += 1

        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT or (x, y) in self.paredes:
            return False

        self.agente_pos = (x, y)
        if self.agente_pos in self.frutas:
            self.frutas.remove(self.agente_pos)
            if not self.frutas:
                print("\n✨ ¡Ganaste! Recolectaste todas las frutas.\n")
                return True
        elif self.agente_pos in self.venenos:
            print("\n☠️ ¡Oh no! Tocaste un veneno.\n")
            return True
        return False

    def dibujar(self, pantalla, modo, cursor_pos, img_fruta, img_veneno, img_pared, img_agente, _):
        pantalla.fill(COLOR_FONDO)

        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0]*CELL_SIZE, fruta[1]*CELL_SIZE))
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0]*CELL_SIZE, veneno[1]*CELL_SIZE))
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0]*CELL_SIZE, pared[1]*CELL_SIZE))

        pantalla.blit(img_agente, (self.agente_pos[0]*CELL_SIZE, self.agente_pos[1]*CELL_SIZE))

        if modo == "SETUP":
            cursor_rect = pygame.Rect(cursor_pos[0]*CELL_SIZE, cursor_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        font = pygame.font.Font(None, 30)
        pantalla.blit(font.render(f"Modo: {modo}", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 5))
        pantalla.blit(font.render("F: Fruta, V: Veneno, W: Pared, C: Limpiar, H: Jugar", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 30))
        pantalla.blit(font.render("Descubre los controles ocultos usando letras/números", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 55))

def cargar_imagen(nombre, fallback_color):
    try:
        ruta = os.path.join(os.path.dirname(__file__), nombre)
        img = pygame.image.load(ruta).convert_alpha()
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except:
        s = pygame.Surface((CELL_SIZE, CELL_SIZE))
        s.fill(fallback_color)
        return s

def generar_controles_aleatorios():
    teclas = random.sample(TECLAS_VALIDAS, 4)
    acciones = [0, 1, 2, 3]  # Arriba, abajo, izq, der
    random.shuffle(acciones)
    return dict(zip(teclas, acciones))

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
    pygame.display.set_caption("Modo Humano Aleatorio - Come Frutas")

    entorno = EntornoHumano()
    cursor_pos = [0, 0]
    modo = "SETUP"
    mapeo_controles = generar_controles_aleatorios()

    img_fruta = cargar_imagen("fruta.png", (40, 200, 40))
    img_veneno = cargar_imagen("veneno.png", (255, 50, 50))
    img_pared = cargar_imagen("pared.jpg", (80, 80, 80))
    img_agente = cargar_imagen("agente.png", (60, 100, 255))

    reloj = pygame.time.Clock()
    corriendo = True

    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_s:
                    modo = "SETUP"
                elif evento.key == pygame.K_h:
                    modo = "HUMANO"
                    entorno.reset()
                    mapeo_controles = generar_controles_aleatorios()

                if modo == "SETUP":
                    if evento.key == pygame.K_UP: cursor_pos[1] = max(0, cursor_pos[1]-1)
                    elif evento.key == pygame.K_DOWN: cursor_pos[1] = min(GRID_HEIGHT-1, cursor_pos[1]+1)
                    elif evento.key == pygame.K_LEFT: cursor_pos[0] = max(0, cursor_pos[0]-1)
                    elif evento.key == pygame.K_RIGHT: cursor_pos[0] = min(GRID_WIDTH-1, cursor_pos[0]+1)
                    pos = tuple(cursor_pos)
                    if evento.key == pygame.K_f: entorno.frutas.symmetric_difference_update({pos}); entorno.venenos.discard(pos); entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_v: entorno.venenos.symmetric_difference_update({pos}); entorno.frutas.discard(pos); entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_w: entorno.paredes.symmetric_difference_update({pos}); entorno.frutas.discard(pos); entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_c: entorno.limpiar()

                elif modo == "HUMANO":
                    if evento.key in mapeo_controles:
                        accion = mapeo_controles[evento.key]
                        terminado = entorno.step(accion)
                        if terminado:
                            modo = "SETUP"

        entorno.dibujar(pantalla, modo, cursor_pos, img_fruta, img_veneno, img_pared, img_agente, mapeo_controles)
        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()