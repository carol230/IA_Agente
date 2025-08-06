"""
Demostración interactiva de agente entrenado por aprendizaje por imitación.

Este módulo proporciona una interfaz gráfica para demostrar el comportamiento
de un agente que ha aprendido por imitación de datos expertos. Permite al
usuario diseñar niveles y observar cómo el agente navega utilizando la
política aprendida mediante redes neuronales convolucionales.

Características:
    - Modo configuración: Diseño interactivo de niveles
    - Modo juego: Demostración automática del agente entrenado
    - Interfaz visual: Pygame con sprites y feedback en tiempo real
    - Carga de modelos: Integración con modelos PyTorch pre-entrenados

Constantes:
    GRID_WIDTH, GRID_HEIGHT: Dimensiones del entorno (5x5)
    CELL_SIZE: Tamaño de cada celda en píxeles (120px)
    SCREEN_WIDTH, SCREEN_HEIGHT: Dimensiones de la ventana
    COLOR_*: Esquema de colores para la interfaz

Clases:
    EntornoGrid: Entorno de demostración con funcionalidades completas
"""
import pygame
import numpy as np
import os
import time
from agent import Agent

# Configuración del entorno y pantalla
GRID_WIDTH = 5
GRID_HEIGHT = 5
CELL_SIZE = 120
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Esquema de colores para interfaz oscura
COLOR_FONDO = (25, 25, 25)      # Fondo principal oscuro
COLOR_LINEAS = (40, 40, 40)     # Líneas de grid sutiles
COLOR_CURSOR = (255, 255, 0)    # Cursor amarillo brillante
COLOR_TEXTO = (230, 230, 230)   # Texto claro para legibilidad


class EntornoGrid:
    """
    Entorno de demostración para agente entrenado por imitación.
    
    Implementa un entorno de grid completo con capacidades de configuración
    interactiva y simulación de episodios. Diseñado para demostrar el
    comportamiento aprendido del agente en diferentes escenarios.
    
    Funcionalidades:
        - Configuración manual de elementos (frutas, venenos, paredes)
        - Simulación de episodios con agente automático
        - Sistema de recompensas completo para feedback
        - Renderizado visual con sprites
        - Detección de colisiones y condiciones de terminación
    
    Attributes:
        size (int): Tamaño del grid (5x5)
        agent_pos (tuple): Posición actual del agente (fila, columna)
        frutas (set): Conjunto de posiciones de frutas
        venenos (set): Conjunto de posiciones de venenos
        paredes (set): Conjunto de posiciones de paredes
    """
    def __init__(self):
        """
        Inicializa el entorno con configuración por defecto.
        
        Establece grid vacío con agente en posición (0,0) y
        conjuntos vacíos para elementos del entorno.
        """
        self.size = GRID_WIDTH
        self.agent_pos = (0, 0)
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset_a_configuracion_inicial(self):
        """
        Reinicia el agente a la posición inicial del episodio.
        
        Coloca al agente en (0,0) manteniendo la configuración actual
        del entorno. Utilizado al iniciar nuevos episodios de demostración.
        
        Returns:
            np.ndarray: Estado inicial del entorno con forma (3, size, size)
        """
        self.agent_pos = (0, 0)
        return self.get_state()

    def limpiar_entorno(self):
        """
        Elimina todos los elementos del entorno.
        
        Limpia completamente frutas, venenos y paredes, dejando
        un grid vacío para nueva configuración. El agente mantiene
        su posición actual.
        """
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        """
        Ejecuta una acción del agente y actualiza el estado del entorno.
        
        Procesa el movimiento del agente, verifica colisiones y calcula
        recompensas según las interacciones con elementos del entorno.
        Implementa la lógica completa de simulación para demostración.
        
        Sistema de recompensas:
            - Movimiento normal: -0.05 (costo por paso)
            - Movimiento inválido: -0.1 (penalización)
            - Fruta recolectada: +1.0 (objetivo positivo)
            - Todas las frutas: +10.0 adicional (victoria)
            - Veneno tocado: -10.0 (penalización grave)
        
        Args:
            accion (int): Acción a ejecutar
                         0 = Arriba (decrementar fila)
                         1 = Abajo (incrementar fila)
                         2 = Izquierda (decrementar columna)
                         3 = Derecha (incrementar columna)
        
        Returns:
            tuple: (nuevo_estado, recompensa, terminado)
                - nuevo_estado (np.ndarray): Estado resultante (3, size, size)
                - recompensa (float): Recompensa por la acción ejecutada
                - terminado (bool): True si episodio terminó, False en caso contrario
        """
        # Calcular nueva posición basada en la acción
        fila, col = self.agent_pos
        if accion == 0:
            fila -= 1    # Arriba
        elif accion == 1:
            fila += 1    # Abajo
        elif accion == 2:
            col -= 1     # Izquierda
        elif accion == 3:
            col += 1     # Derecha

        # Verificar límites del entorno y colisiones con paredes
        if (
            fila < 0
            or fila >= GRID_HEIGHT
            or col < 0
            or col >= GRID_WIDTH
            or (fila, col) in self.paredes
        ):
            # Movimiento inválido: mantener posición y penalizar
            return self.get_state(), -0.1, False

        # Movimiento válido: actualizar posición
        self.agent_pos = (fila, col)
        recompensa = -0.05  # Costo base por movimiento
        terminado = False

        # Procesar interacciones con elementos del entorno
        if self.agent_pos in self.venenos:
            # Veneno tocado: penalización grave y reset a inicio
            recompensa = -10.0
            self.agent_pos = (0, 0)
        elif self.agent_pos in self.frutas:
            # Fruta recolectada: recompensa positiva
            recompensa = 1.0
            self.frutas.remove(self.agent_pos)
            # Verificar victoria (todas las frutas recolectadas)
            if not self.frutas:
                recompensa += 10.0  # Bonus por completar nivel
                terminado = True
                self.agent_pos = (0, 0)  # Reset a posición inicial

        return self.get_state(), recompensa, terminado

    def get_state(self):
        """
        Genera representación visual del estado actual del entorno.
        
        Crea tensor 3D donde cada canal representa un tipo de elemento,
        compatible con la arquitectura CNN del agente entrenado.
        
        Estructura de canales:
            - Canal 0: Posición del agente (binario)
            - Canal 1: Posiciones de frutas (binario)
            - Canal 2: Posiciones de venenos (binario)
        
        Returns:
            np.ndarray: Estado con forma (3, size, size) y dtype float32
                       Valores 1.0 indican presencia, 0.0 ausencia
        
        Note:
            Las paredes no se incluyen en el estado ya que el agente
            entrenado no las consideraba en los datos de demostración.
        """
        # Inicializar tensor de estado
        estado = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Canal 0: Posición del agente
        estado[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Canal 1: Posiciones de frutas
        for fruta in self.frutas:
            estado[1, fruta[0], fruta[1]] = 1.0
        
        # Canal 2: Posiciones de venenos
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
        """
        Renderiza el estado completo del entorno con interfaz de usuario.
        
        Dibuja todos los elementos visuales del entorno, grid de navegación,
        cursor de configuración e información de controles. Proporciona
        feedback visual completo para ambos modos de operación.
        
        Args:
            pantalla (pygame.Surface): Superficie donde renderizar
            modo_juego (str): Modo actual ("SETUP" o "PLAYING")
            cursor_pos (tuple): Posición del cursor en modo configuración
            img_fruta (pygame.Surface): Sprite de las frutas
            img_veneno (pygame.Surface): Sprite de los venenos
            img_pared (pygame.Surface): Sprite de las paredes
            img_agente (pygame.Surface): Sprite del agente
        
        Note:
            Renderiza en orden específico para evitar superposiciones:
            fondo → grid → paredes → frutas → venenos → agente → cursor → UI
        """
        # Limpiar pantalla con fondo oscuro
        pantalla.fill(COLOR_FONDO)
        
        # Dibujar líneas del grid para navegación visual
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Renderizar elementos del entorno (orden: paredes → frutas → venenos)
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0] * CELL_SIZE, pared[1] * CELL_SIZE))
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0] * CELL_SIZE, fruta[1] * CELL_SIZE))
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0] * CELL_SIZE, veneno[1] * CELL_SIZE))

        # Dibujar agente (siempre en primer plano)
        pantalla.blit(
            img_agente, (self.agent_pos[0] * CELL_SIZE, self.agent_pos[1] * CELL_SIZE)
        )

        # Mostrar cursor en modo configuración
        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        # Renderizar información de interfaz
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
    """
    Función principal de la demostración interactiva del agente por imitación.
    
    Inicializa la interfaz gráfica y gestiona el bucle principal que permite
    alternar entre modo configuración (diseño de niveles) y modo demostración
    (agente automático). Proporciona una experiencia completa para evaluar
    el rendimiento del agente entrenado.
    
    Flujo de la aplicación:
        1. Inicialización de Pygame y recursos
        2. Carga del modelo entrenado
        3. Bucle principal con dos modos:
           - SETUP: Configuración manual de niveles
           - PLAYING: Demostración automática del agente
        4. Manejo de eventos y renderizado en tiempo real
    
    Controles disponibles:
        Modo SETUP:
            - Flechas: Mover cursor de configuración
            - F: Colocar/quitar fruta
            - V: Colocar/quitar veneno
            - W: Colocar/quitar pared
            - C: Limpiar entorno completamente
            - P: Iniciar demostración automática
        
        Modo PLAYING:
            - S: Volver a modo configuración
            - Agente se mueve automáticamente cada 0.1 segundos
    
    Note:
        Requiere modelo entrenado en "IMITACION/imitacion_model.pth"
        y sprites en directorio padre (../fruta.png, etc.)
    """
    # Inicializar Pygame y configurar ventana
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente por Imitación - Come Frutas 🍓")

    def cargar_img(nombre, color_fallback):
        """
        Función auxiliar para carga robusta de sprites.
        
        Intenta cargar imagen desde archivo, si falla crea superficie
        de color sólido como respaldo para mantener funcionalidad.
        
        Args:
            nombre (str): Nombre del archivo de imagen
            color_fallback (tuple): Color RGB de respaldo
        
        Returns:
            pygame.Surface: Sprite cargado o superficie de color
        """
        try:
            ruta = os.path.join(os.path.dirname(__file__), nombre)
            img = pygame.image.load(ruta).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except:
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.fill(color_fallback)
            return surf

    # Cargar sprites con colores de respaldo
    img_fruta = cargar_img("../fruta.png", (0, 255, 0))        # Verde
    img_veneno = cargar_img("../veneno.png", (255, 0, 0))      # Rojo
    img_pared = cargar_img("../pared.png", (100, 100, 100))    # Gris
    img_agente = cargar_img("../agente.png", (0, 0, 255))      # Azul

    # Inicializar entorno y agente
    entorno = EntornoGrid()
    agente = Agent()
    agente.load_model("IMITACION/imitacion_model.pth")

    # Variables de estado de la aplicación
    cursor_pos = [0, 0]
    modo_juego = "SETUP"
    reloj = pygame.time.Clock()
    corriendo = True

    # Bucle principal de la aplicación
    while corriendo:
        # Procesar eventos de entrada
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                # Cambio de modos globales
                if evento.key == pygame.K_p:
                    print("--- MODO JUEGO ---")
                    entorno.reset_a_configuracion_inicial()
                    modo_juego = "PLAYING"
                    time.sleep(0.5)  # Pausa para visibilidad del cambio

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                # Controles específicos del modo SETUP
                if modo_juego == "SETUP":
                    # Navegación del cursor con flechas
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_HEIGHT - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_WIDTH - 1, cursor_pos[0] + 1)

                    # Colocación/eliminación de elementos
                    pos = tuple(cursor_pos)
                    if evento.key == pygame.K_f:
                        # Toggle fruta: agregar/quitar y limpiar otros elementos
                        if pos in entorno.frutas:
                            entorno.frutas.remove(pos)
                        else:
                            entorno.frutas.add(pos)
                            entorno.venenos.discard(pos)
                            entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_v:
                        # Toggle veneno: agregar/quitar y limpiar otros elementos
                        if pos in entorno.venenos:
                            entorno.venenos.remove(pos)
                        else:
                            entorno.venenos.add(pos)
                            entorno.frutas.discard(pos)
                            entorno.paredes.discard(pos)
                    elif evento.key == pygame.K_w:
                        # Toggle pared: agregar/quitar y limpiar otros elementos
                        if pos in entorno.paredes:
                            entorno.paredes.remove(pos)
                        else:
                            entorno.paredes.add(pos)
                            entorno.frutas.discard(pos)
                            entorno.venenos.discard(pos)
                    elif evento.key == pygame.K_c:
                        # Limpiar entorno completamente
                        print("--- LIMPIANDO ENTORNO ---")
                        entorno.limpiar_entorno()

        # Lógica del modo PLAYING (agente automático)
        if modo_juego == "PLAYING":
            # Obtener estado actual y decidir acción
            estado = entorno.get_state()
            accion = agente.choose_action(estado)
            # Ejecutar acción y verificar terminación
            _, _, terminado = entorno.step(accion)
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
            time.sleep(0.1)  # Velocidad de demostración controlada

        # Renderizado del estado actual
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
        reloj.tick(60)  # 60 FPS para fluidez visual

    # Limpiar recursos al salir
    pygame.quit()


if __name__ == "__main__":
    main()
