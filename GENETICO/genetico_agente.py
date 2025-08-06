"""
Demostrador interactivo para agentes entrenados con algoritmos genéticos.

Este módulo implementa una interfaz gráfica completa que permite:
1. Configurar escenarios personalizados con frutas, venenos y paredes
2. Observar el comportamiento de un agente genético entrenado
3. Interactuar en tiempo real con controles de teclado
4. Visualizar el rendimiento del agente en diferentes configuraciones

Características principales:
- Modo Setup: Configuración manual del entorno
- Modo Playing: Demostración del agente en acción
- Controles intuitivos con teclado
- Gráficos mejorados con sprites
- Interfaz informativa con instrucciones

El agente carga pesos previamente evolucionados y demuestra su comportamiento
determinístico en los escenarios configurados por el usuario.

"""

import pygame
import numpy as np
import os
import time
from agent_ga import Agent

# CONFIGURACIÓN VISUAL Y DIMENSIONES
"""
Constantes que definen la apariencia y dimensiones de la interfaz gráfica.
"""
GRID_WIDTH = 5          # Ancho de la cuadrícula en celdas
GRID_HEIGHT = 5         # Alto de la cuadrícula en celdas  
CELL_SIZE = 120         # Tamaño de cada celda en píxeles (más grande que en DQN)
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE    # Ancho total de la ventana (600px)
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE  # Alto total del área de juego (600px)

# PALETA DE COLORES PROFESIONAL
"""
Esquema de colores dark theme para una interfaz moderna y profesional.
"""
COLOR_FONDO = (25, 25, 25)      # Gris muy oscuro para el fondo
COLOR_LINEAS = (40, 40, 40)     # Gris oscuro para líneas de cuadrícula
COLOR_CURSOR = (255, 255, 0)    # Amarillo brillante para el cursor de selección
COLOR_TEXTO = (230, 230, 230)   # Gris claro para texto legible


class EntornoGrid:
    """
    Entorno de cuadrícula personalizado para la demostración del agente genético.
    
    Esta clase maneja la lógica del juego y la configuración del entorno,
    incluyendo la colocación de elementos y la simulación de la interacción
    del agente. Incluye características adicionales como paredes que no
    están presentes en los entornos de entrenamiento básicos.
    
    Características especiales:
    - Soporte para paredes como obstáculos
    - Interfaz de configuración manual
    - Reset automático en condiciones de terminación
    - Estado visual compatible con el agente entrenado
    
    Attributes:
        size (int): Tamaño de la cuadrícula
        agent_pos (tuple): Posición actual del agente (fila, columna)
        frutas (set): Conjunto de posiciones con frutas
        venenos (set): Conjunto de posiciones con venenos  
        paredes (set): Conjunto de posiciones con paredes (obstáculos)
    """
    def __init__(self):
        """
        Inicializa el entorno con configuración vacía.
        
        Todos los conjuntos de elementos comienzan vacíos, permitiendo
        al usuario configurar el escenario manualmente.
        """
        self.size = GRID_WIDTH
        self.agent_pos = (0, 0)    # Agente siempre inicia en esquina superior izquierda
        self.frutas = set()        # Conjunto de posiciones de frutas
        self.venenos = set()       # Conjunto de posiciones de venenos
        self.paredes = set()       # Conjunto de posiciones de paredes

    def reset_a_configuracion_inicial(self):
        """
        Resetea el agente a la posición inicial sin modificar el entorno.
        
        Utilizado al inicio de cada demostración para colocar al agente
        en la posición de partida estándar (0,0) manteniendo la configuración
        de frutas, venenos y paredes establecida por el usuario.
        
        Returns:
            np.array: Estado inicial del entorno después del reset
        """
        self.agent_pos = (0, 0)
        return self.get_state()

    def limpiar_entorno(self):
        """
        Elimina todos los elementos del entorno (frutas, venenos, paredes).
        
        Función de utilidad para resetear completamente el escenario,
        permitiendo al usuario comenzar con una cuadrícula vacía.
        El agente permanece en su posición actual.
        """
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        """
        Ejecuta una acción del agente en el entorno de demostración.
        
        Implementa la lógica del juego incluyendo movimiento, colisiones con
        paredes, interacción con elementos del entorno y cálculo de recompensas.
        Incluye características especiales como paredes que bloquean el movimiento.
        
        Diferencias con el entorno de entrenamiento:
        - Incluye paredes como obstáculos
        - Movimientos inválidos dan recompensa negativa
        - Reset automático al completar nivel
        
        Args:
            accion (int): Acción a ejecutar:
                         0 = Arriba (decrementar fila)
                         1 = Abajo (incrementar fila) 
                         2 = Izquierda (decrementar columna)
                         3 = Derecha (incrementar columna)
        
        Returns:
            tuple: (estado, recompensa, terminado)
                - estado (np.array): Nuevo estado del entorno
                - recompensa (float): Recompensa obtenida
                - terminado (bool): Si el episodio ha terminado
        """
        # Calcular nueva posición basada en la acción
        fila, col = self.agent_pos
        if accion == 0:     # Arriba
            fila -= 1
        elif accion == 1:   # Abajo
            fila += 1
        elif accion == 2:   # Izquierda
            col -= 1
        elif accion == 3:   # Derecha
            col += 1

        # Verificar colisiones: límites del tablero o paredes
        if (
            fila < 0
            or fila >= GRID_HEIGHT
            or col < 0
            or col >= GRID_WIDTH
            or (fila, col) in self.paredes
        ):
            # Movimiento inválido: pequeña penalización, posición no cambia
            return self.get_state(), -0.1, False

        # Movimiento válido: actualizar posición
        self.agent_pos = (fila, col)
        recompensa = -0.05    # Costo base del movimiento
        terminado = False

        # Verificar interacciones con elementos del entorno
        if self.agent_pos in self.venenos:
            # Veneno tocado: penalización severa y reset a inicio
            recompensa = -10.0
            self.agent_pos = (0, 0)
        elif self.agent_pos in self.frutas:
            # Fruta recogida: recompensa positiva
            recompensa = 1.0
            self.frutas.remove(self.agent_pos)
            
            # Verificar si se completó el nivel
            if not self.frutas:
                recompensa += 10.0    # Bonus por completar
                terminado = True
                self.agent_pos = (0, 0)  # Reset para próxima demostración

        return self.get_state(), recompensa, terminado

    def get_state(self):
        """
        Genera la representación del estado compatible con el agente entrenado.
        
        Crea una representación de 3 canales idéntica a la utilizada durante
        el entrenamiento, asegurando compatibilidad con los pesos evolucionados.
        Las paredes no se incluyen en el estado ya que el agente original
        no fue entrenado con ellas.
        
        Returns:
            np.array: Estado del entorno de forma (3, size, size):
                     - Canal 0: Posición del agente
                     - Canal 1: Posiciones de frutas
                     - Canal 2: Posiciones de venenos
        """
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
        Renderiza el estado completo del entorno en la pantalla.
        
        Dibuja todos los elementos visuales incluyendo cuadrícula, sprites,
        cursor de selección (en modo setup) e interfaz de usuario con
        controles e información del modo actual.
        
        Args:
            pantalla (pygame.Surface): Superficie donde dibujar
            modo_juego (str): Modo actual ("SETUP" o "PLAYING")
            cursor_pos (tuple): Posición del cursor en modo setup
            img_fruta (pygame.Surface): Sprite de la fruta
            img_veneno (pygame.Surface): Sprite del veneno
            img_pared (pygame.Surface): Sprite de la pared
            img_agente (pygame.Surface): Sprite del agente
        """
        # Limpiar pantalla con color de fondo
        pantalla.fill(COLOR_FONDO)
        
        # Dibujar cuadrícula de referencia
        # Líneas verticales
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        # Líneas horizontales
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Dibujar elementos del entorno (orden importante para superposición correcta)
        # 1. Paredes (fondo)
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0] * CELL_SIZE, pared[1] * CELL_SIZE))
        
        # 2. Frutas
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0] * CELL_SIZE, fruta[1] * CELL_SIZE))
        
        # 3. Venenos
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0] * CELL_SIZE, veneno[1] * CELL_SIZE))

        # 4. Agente (primer plano)
        pantalla.blit(
            img_agente, (self.agent_pos[0] * CELL_SIZE, self.agent_pos[1] * CELL_SIZE)
        )

        # 5. Cursor de selección (solo en modo setup)
        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        # Dibujar interfaz de usuario en la parte inferior
        font = pygame.font.Font(None, 24)
        
        # Información del modo actual
        texto_modo = font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO)
        
        # Controles disponibles
        controles1 = font.render(
            "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar", True, COLOR_TEXTO
        )
        controles2 = font.render("P=Jugar, S=Setup", True, COLOR_TEXTO)
        
        # Posicionar texto en la parte inferior
        pantalla.blit(texto_modo, (10, SCREEN_HEIGHT + 5))
        pantalla.blit(controles1, (10, SCREEN_HEIGHT + 30))
        pantalla.blit(controles2, (10, SCREEN_HEIGHT + 55))


def main():
    """
    Función principal que ejecuta la aplicación de demostración.
    
    Inicializa Pygame, carga recursos gráficos, configura el agente genético
    entrenado y ejecuta el bucle principal de la aplicación. Maneja dos modos
    principales: configuración manual y demostración automática.
    
    Flujo de ejecución:
    1. Inicialización de Pygame y recursos
    2. Carga del agente entrenado
    3. Bucle principal con manejo de eventos
    4. Renderizado continuo
    5. Limpieza al salir
    """
    # INICIALIZACIÓN DE PYGAME
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente Genético - Come Frutas 🍓")

    # FUNCIÓN AUXILIAR PARA CARGA DE IMÁGENES
    def cargar_img(nombre, color_fallback):
        """
        Carga una imagen con fallback a color sólido si falla.
        
        Args:
            nombre (str): Nombre del archivo de imagen
            color_fallback (tuple): Color RGB de respaldo
            
        Returns:
            pygame.Surface: Superficie escalada al tamaño de celda
        """
        try:
            ruta = os.path.join(os.path.dirname(__file__), nombre)
            img = pygame.image.load(ruta).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except:
            # Fallback: crear superficie de color sólido
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.fill(color_fallback)
            return surf

    # CARGA DE RECURSOS GRÁFICOS
    img_fruta = cargar_img("../fruta.png", (0, 255, 0))      # Verde si no hay imagen
    img_veneno = cargar_img("../veneno.png", (255, 0, 0))    # Rojo si no hay imagen  
    img_pared = cargar_img("../pared.png", (100, 100, 100)) # Gris si no hay imagen
    img_agente = cargar_img("../agente.png", (0, 0, 255))   # Azul si no hay imagen

    # INICIALIZACIÓN DEL ENTORNO Y AGENTE
    entorno = EntornoGrid()
    agente = Agent()
    
    # Cargar agente entrenado con algoritmos genéticos
    agente.load_genes("GENETICO/best_agent_genes.pth")

    # VARIABLES DE ESTADO DE LA APLICACIÓN
    cursor_pos = [0, 0]        # Posición del cursor en modo setup
    modo_juego = "SETUP"       # Modo inicial: configuración
    reloj = pygame.time.Clock() # Control de FPS
    corriendo = True           # Flag de control del bucle principal

    # BUCLE PRINCIPAL DE LA APLICACIÓN
    while corriendo:
        # MANEJO DE EVENTOS
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            # EVENTOS DE TECLADO
            if evento.type == pygame.KEYDOWN:
                # CONTROLES GLOBALES (disponibles en ambos modos)
                if evento.key == pygame.K_p:
                    print("--- INICIANDO MODO JUEGO ---")
                    entorno.reset_a_configuracion_inicial()
                    modo_juego = "PLAYING"
                    time.sleep(0.5)  # Pausa para transición visual

                elif evento.key == pygame.K_s:
                    print("--- INICIANDO MODO SETUP ---")
                    modo_juego = "SETUP"

                # CONTROLES ESPECÍFICOS DEL MODO SETUP
                if modo_juego == "SETUP":
                    # Navegación con flechas del cursor
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
                    
                    # F = Toggle Fruta
                    if evento.key == pygame.K_f:
                        if pos in entorno.frutas:
                            entorno.frutas.remove(pos)
                            print(f"Fruta eliminada en {pos}")
                        else:
                            entorno.frutas.add(pos)
                            entorno.venenos.discard(pos)    # Remover otros elementos
                            entorno.paredes.discard(pos)
                            print(f"Fruta colocada en {pos}")
                    
                    # V = Toggle Veneno
                    elif evento.key == pygame.K_v:
                        if pos in entorno.venenos:
                            entorno.venenos.remove(pos)
                            print(f"Veneno eliminado en {pos}")
                        else:
                            entorno.venenos.add(pos)
                            entorno.frutas.discard(pos)     # Remover otros elementos
                            entorno.paredes.discard(pos)
                            print(f"Veneno colocado en {pos}")
                    
                    # W = Toggle Pared
                    elif evento.key == pygame.K_w:
                        if pos in entorno.paredes:
                            entorno.paredes.remove(pos)
                            print(f"Pared eliminada en {pos}")
                        else:
                            entorno.paredes.add(pos)
                            entorno.frutas.discard(pos)     # Remover otros elementos
                            entorno.venenos.discard(pos)
                            print(f"Pared colocada en {pos}")
                    
                    # C = Limpiar todo
                    elif evento.key == pygame.K_c:
                        print("--- LIMPIANDO ENTORNO COMPLETO ---")
                        entorno.limpiar_entorno()

        # LÓGICA DEL MODO PLAYING (DEMOSTRACIÓN DEL AGENTE)
        if modo_juego == "PLAYING":
            # El agente toma decisiones automáticamente
            estado = entorno.get_state()
            accion = agente.choose_action(estado)
            _, _, terminado = entorno.step(accion)
            
            # Verificar si el episodio terminó
            if terminado:
                print("🏆 ¡Agente completó el nivel! Volviendo a modo SETUP.")
                modo_juego = "SETUP"
            
            # Pausa para visualización clara del movimiento
            time.sleep(0.1)

        # RENDERIZADO (COMÚN PARA AMBOS MODOS)
        # Crear superficie temporal para el contenido completo
        pantalla_con_info = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
        pantalla_con_info.fill(COLOR_FONDO)
        
        # Dibujar entorno y elementos
        entorno.dibujar(
            pantalla_con_info,
            modo_juego,
            tuple(cursor_pos),
            img_fruta,
            img_veneno,
            img_pared,
            img_agente,
        )
        
        # Copiar a pantalla principal y actualizar
        pantalla.blit(pantalla_con_info, (0, 0))
        pygame.display.flip()
        
        # Controlar FPS
        reloj.tick(60)

    # LIMPIEZA AL SALIR
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa.
    
    Ejecuta la función main() solo si este archivo se ejecuta directamente.
    Incluye mensaje de bienvenida con instrucciones básicas.
    """
    print("=" * 60)
    print("🧬 DEMOSTRADOR DE AGENTE GENÉTICO 🧬")
    print("=" * 60)
    print("CONTROLES:")
    print("🎮 GLOBALES:")
    print("  P - Iniciar modo Playing (demostración)")
    print("  S - Cambiar a modo Setup (configuración)")
    print()
    print("⚙️  MODO SETUP:")
    print("  ⬆️⬇️⬅️➡️ - Mover cursor")
    print("  F - Toggle Fruta")
    print("  V - Toggle Veneno") 
    print("  W - Toggle Pared")
    print("  C - Limpiar entorno")
    print()
    print("🤖 MODO PLAYING:")
    print("  El agente toma control automáticamente")
    print("  Observa el comportamiento evolucionado")
    print()
    print("¡Configura un escenario y observa la inteligencia artificial!")
    print("=" * 60)
    
    main()
