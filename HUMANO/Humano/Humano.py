"""
Modo de juego humano con controles aleatorios - Come Frutas.

Este módulo implementa una versión jugable del entorno donde un humano puede
controlar el agente directamente. La característica única es que los controles
de movimiento se asignan aleatoriamente cada vez que se inicia una partida,
añadiendo un elemento de desafío y adaptabilidad.

Características principales:
- Modo Setup: Configuración manual del escenario
- Modo Humano: Control directo del agente por el jugador
- Controles aleatorios: Mapeo aleatorio de teclas a movimientos
- Interfaz intuitiva: Gráficos y feedback visual
- Desafío adaptativo: Cada partida requiere aprender nuevos controles

Propósito educativo:
- Comparar rendimiento humano vs. IA
- Experimentar la dificultad de adaptación a controles cambiantes
- Entender la importancia de la consistencia en interfaces
- Apreciar la flexibilidad del aprendizaje humano

Autor: [Tu nombre]
Fecha: Agosto 2025
"""

import pygame
import os
import random
import string

# CONFIGURACIÓN DEL ENTORNO VISUAL
"""
Parámetros visuales y dimensiones de la interfaz de juego.
Utiliza celdas más grandes (120px) para mejor visibilidad durante el juego manual.
"""
GRID_WIDTH = 5              # Ancho de la cuadrícula en celdas
GRID_HEIGHT = 5             # Alto de la cuadrícula en celdas
CELL_SIZE = 120             # Tamaño de cada celda en píxeles (mayor para juego manual)
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE    # Ancho total de la ventana (600px)
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE  # Alto del área de juego (600px)

# PALETA DE COLORES CONSISTENTE
"""
Esquema de colores oscuro profesional, consistente con otros módulos del proyecto.
"""
COLOR_FONDO = (25, 25, 25)      # Gris muy oscuro para el fondo
COLOR_LINEAS = (40, 40, 40)     # Gris oscuro para líneas de cuadrícula
COLOR_CURSOR = (255, 255, 0)    # Amarillo brillante para cursor de selección
COLOR_TEXTO = (230, 230, 230)   # Gris claro para texto legible

# SISTEMA DE CONTROLES ALEATORIOS
"""
Genera un conjunto de teclas válidas para asignación aleatoria de controles.
Se evitan teclas especiales para prevenir conflictos con funciones del sistema.
"""
TECLAS_VALIDAS = [getattr(pygame, f"K_{c}") for c in string.ascii_lowercase + string.digits]

class EntornoHumano:
    """
    Entorno de juego optimizado para control humano directo.
    
    Esta clase maneja la lógica del juego cuando un humano controla el agente,
    incluyendo movimiento, colisiones, recolección de objetos y condiciones
    de victoria/derrota. Se enfoca en proporcionar feedback inmediato y
    una experiencia de juego fluida.
    
    Diferencias con entornos de IA:
    - Feedback inmediato con mensajes en consola
    - Lógica de juego simplificada (sin recompensas numéricas)
    - Terminación inmediata en victoria/derrota
    - Controles responsivos para jugabilidad humana
    
    Attributes:
        agente_pos (tuple): Posición actual del agente (x, y)
        frutas (set): Conjunto de posiciones con frutas
        venenos (set): Conjunto de posiciones con venenos
        paredes (set): Conjunto de posiciones con paredes/obstáculos
    """
    def __init__(self):
        """
        Inicializa el entorno con configuración vacía.
        
        El agente comienza en la esquina superior izquierda (0,0) y todos
        los conjuntos de elementos están vacíos, permitiendo configuración manual.
        """
        self.agente_pos = (0, 0)    # Posición inicial del agente
        self.frutas = set()         # Conjunto de posiciones de frutas
        self.venenos = set()        # Conjunto de posiciones de venenos
        self.paredes = set()        # Conjunto de posiciones de paredes

    def reset(self):
        """
        Resetea la posición del agente al inicio del juego.
        
        Coloca al agente en la posición inicial (0,0) sin modificar
        la configuración del escenario. Utilizado al comenzar una nueva partida.
        """
        self.agente_pos = (0, 0)

    def limpiar(self):
        """
        Elimina todos los elementos del entorno.
        
        Limpia frutas, venenos y paredes del escenario, dejando una
        cuadrícula vacía para configuración desde cero.
        """
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        """
        Ejecuta una acción del jugador humano en el entorno.
        
        Procesa el movimiento del agente, verifica colisiones y maneja
        las interacciones con elementos del entorno. Proporciona feedback
        inmediato al jugador mediante mensajes en consola.
        
        Args:
            accion (int): Dirección de movimiento:
                         0 = Arriba (decrementar y)
                         1 = Abajo (incrementar y)
                         2 = Izquierda (decrementar x)
                         3 = Derecha (incrementar x)
        
        Returns:
            bool: True si el juego terminó (victoria o derrota), False si continúa
        """
        # Calcular nueva posición basada en la acción
        x, y = self.agente_pos
        if accion == 0:     # Arriba
            y -= 1
        elif accion == 1:   # Abajo
            y += 1
        elif accion == 2:   # Izquierda
            x -= 1
        elif accion == 3:   # Derecha
            x += 1

        # Verificar colisiones: límites del tablero o paredes
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT or (x, y) in self.paredes:
            # Movimiento inválido: no actualizar posición
            return False

        # Movimiento válido: actualizar posición del agente
        self.agente_pos = (x, y)
        
        # Verificar interacciones con elementos del entorno
        if self.agente_pos in self.frutas:
            # Fruta recogida: eliminar del conjunto
            self.frutas.remove(self.agente_pos)
            
            # Verificar condición de victoria
            if not self.frutas:
                print("\n✨ ¡Ganaste! Recolectaste todas las frutas.\n")
                return True  # Juego terminado con éxito
                
        elif self.agente_pos in self.venenos:
            # Veneno tocado: derrota inmediata
            print("\n☠️ ¡Oh no! Tocaste un veneno.\n")
            return True  # Juego terminado con fallo
            
        # Continuar juego
        return False

    def dibujar(self, pantalla, modo, cursor_pos, img_fruta, img_veneno, img_pared, img_agente, _):
        """
        Renderiza el estado completo del entorno con interfaz interactiva.
        
        Dibuja todos los elementos visuales del juego incluyendo grid, objetos
        del entorno y cursor de selección. Proporciona feedback visual para
        la interacción del jugador en diferentes modos (colocación/juego).
        
        Args:
            pantalla (pygame.Surface): Superficie donde renderizar
            modo (str): Modo actual de la interfaz ('frutas', 'venenos', 'paredes', 'jugar')
            cursor_pos (tuple): Posición (x,y) del cursor en coordenadas de grid
            img_fruta (pygame.Surface): Sprite de las frutas
            img_veneno (pygame.Surface): Sprite de los venenos
            img_pared (pygame.Surface): Sprite de las paredes
            img_agente (pygame.Surface): Sprite del agente
            _ : Parámetro no utilizado (compatibilidad de interfaz)
        
        Note:
            Renderiza en orden específico: fondo, grid, objetos, agente, cursor.
            El cursor cambia de color según el modo de colocación activo.
        """
        # Limpiar pantalla con color de fondo
        pantalla.fill(COLOR_FONDO)

        # Dibujar líneas del grid para guía visual
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Dibujar elementos del entorno
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0]*CELL_SIZE, fruta[1]*CELL_SIZE))
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0]*CELL_SIZE, veneno[1]*CELL_SIZE))
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0]*CELL_SIZE, pared[1]*CELL_SIZE))

        # Dibujar agente (jugador) - siempre visible en primer plano
        pantalla.blit(img_agente, (self.agente_pos[0]*CELL_SIZE, self.agente_pos[1]*CELL_SIZE))

        # Dibujar cursor de selección en modo configuración
        if modo == "SETUP":
            cursor_rect = pygame.Rect(cursor_pos[0]*CELL_SIZE, cursor_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        # Renderizar información de interfaz
        font = pygame.font.Font(None, 30)
        pantalla.blit(font.render(f"Modo: {modo}", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 5))
        pantalla.blit(font.render("F: Fruta, V: Veneno, W: Pared, C: Limpiar, H: Jugar", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 30))
        pantalla.blit(font.render("Descubre los controles ocultos usando letras/números", True, COLOR_TEXTO), (10, SCREEN_HEIGHT + 55))

def cargar_imagen(nombre, fallback_color):
    """
    Carga una imagen desde archivo con sistema de respaldo.
    
    Intenta cargar una imagen sprite desde el directorio actual.
    Si la carga falla, crea una superficie de color sólido como respaldo.
    Escala automáticamente al tamaño de celda definido.
    
    Args:
        nombre (str): Nombre del archivo de imagen a cargar
        fallback_color (tuple): Color RGB (r,g,b) para superficie de respaldo
    
    Returns:
        pygame.Surface: Superficie cargada y escalada, o superficie de color
                       si la carga falló
    
    Note:
        Todas las imágenes se escalan a CELL_SIZE x CELL_SIZE píxeles.
        Utiliza convert_alpha() para optimizar el renderizado con transparencia.
    """
    try:
        # Construir ruta completa al archivo de imagen
        ruta = os.path.join(os.path.dirname(__file__), nombre)
        # Cargar imagen con soporte de transparencia
        img = pygame.image.load(ruta).convert_alpha()
        # Escalar a tamaño de celda estándar
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except:
        # Crear superficie de respaldo con color sólido si falla la carga
        s = pygame.Surface((CELL_SIZE, CELL_SIZE))
        s.fill(fallback_color)
        return s

def generar_controles_aleatorios():
    """
    Genera un mapeo aleatorio de teclas para controles de movimiento.
    
    Crea una asignación aleatoria entre teclas del teclado y direcciones
    de movimiento para añadir un elemento de desafío y descubrimiento
    al juego. Los jugadores deben encontrar qué teclas controlan cada dirección.
    
    Returns:
        dict: Mapeo de códigos de tecla pygame a acciones de movimiento:
              {tecla_pygame: accion_int}
              donde accion_int es 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
    
    Note:
        Utiliza teclas alfanuméricas (A-Z, 0-9) para máxima compatibilidad.
        Garantiza que cada dirección tenga exactamente una tecla asignada.
    """
    # Seleccionar 4 teclas aleatorias del conjunto disponible
    teclas = random.sample(TECLAS_VALIDAS, 4)
    # Crear lista de acciones de movimiento
    acciones = [0, 1, 2, 3]  # Arriba, abajo, izquierda, derecha
    # Mezclar aleatoriamente las acciones
    random.shuffle(acciones)
    # Crear diccionario de mapeo tecla->acción
    return dict(zip(teclas, acciones))

def main():
    """
    Función principal del juego en modo humano.
    
    Inicializa Pygame, configura la ventana de juego y ejecuta el bucle
    principal que maneja dos modos: configuración del entorno y juego
    con controles aleatorios. Proporciona una experiencia interactiva
    donde el jugador puede diseñar niveles y luego jugarlos.
    
    Flujo del juego:
        1. Modo SETUP: Colocar frutas, venenos y paredes con el mouse
        2. Modo JUGAR: Controlar agente con teclas aleatorias descubiertas
        3. Victoria: Recolectar todas las frutas
        4. Derrota: Tocar veneno
    
    Controles SETUP:
        - Mouse: Mover cursor
        - F: Colocar fruta
        - V: Colocar veneno  
        - W: Colocar pared
        - C: Limpiar todo
        - H: Iniciar juego
    
    Controles JUGAR:
        - Teclas aleatorias para movimiento (descubrir experimentando)
        - ESC: Volver a configuración
    """
    # Inicializar Pygame y configurar ventana
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
    pygame.display.set_caption("Modo Humano Aleatorio - Come Frutas")

    # Inicializar entorno y variables de estado
    entorno = EntornoHumano()
    cursor_pos = [0, 0]
    modo = "SETUP"  # Modo inicial: configuración del entorno
    mapeo_controles = {}  # Mapeo de teclas aleatorias (generado al jugar)

    # Cargar sprites con colores de respaldo
    img_fruta = cargar_imagen("fruta.png", (40, 200, 40))
    img_veneno = cargar_imagen("veneno.png", (255, 50, 50))
    img_pared = cargar_imagen("pared.jpg", (80, 80, 80))
    img_agente = cargar_imagen("agente.png", (60, 100, 255))

    # Variables de control del juego
    reloj = pygame.time.Clock()
    corriendo = True

    # Bucle principal del juego
    while corriendo:
        # Procesar eventos de entrada
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
                    # Colocación de elementos con teclas específicas
                    pos = tuple(cursor_pos)
                    if evento.key == pygame.K_f: 
                        # F: Colocar/quitar fruta (toggle)
                        entorno.frutas.symmetric_difference_update({pos})
                        entorno.venenos.discard(pos)
                        entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_v: 
                        # V: Colocar/quitar veneno (toggle)
                        entorno.venenos.symmetric_difference_update({pos})
                        entorno.frutas.discard(pos)
                        entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_w: 
                        # W: Colocar/quitar pared (toggle)
                        entorno.paredes.symmetric_difference_update({pos})
                        entorno.frutas.discard(pos)
                        entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_c: 
                        # C: Limpiar todo el entorno
                        entorno.limpiar()

                # Controles específicos del modo HUMANO
                elif modo == "HUMANO":
                    if evento.key in mapeo_controles:
                        # Ejecutar acción de movimiento con tecla aleatoria
                        accion = mapeo_controles[evento.key]
                        terminado = entorno.step(accion)
                        if terminado:
                            # Volver a configuración al terminar el juego
                            modo = "SETUP"

        # Renderizar estado actual del juego
        entorno.dibujar(pantalla, modo, cursor_pos, img_fruta, img_veneno, img_pared, img_agente, mapeo_controles)
        pygame.display.flip()
        reloj.tick(30)

    # Limpiar recursos al salir
    pygame.quit()

if __name__ == '__main__':
    main()