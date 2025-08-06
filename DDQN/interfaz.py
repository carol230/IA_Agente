# ddqn_agente_comefrutas.py (versión integrada con interfaz gráfica completa)
"""
Interfaz gráfica para el entrenamiento y visualización de agentes DDQN.

Este módulo implementa una interfaz gráfica completa usando Pygame que permite:
- Configurar entornos de manera interactiva
- Visualizar el comportamiento del agente entrenado
- Alternar entre modo setup y modo juego
- Gestionar elementos del entorno (frutas, venenos, paredes)

El sistema está diseñado para facilitar la experimentación con diferentes
configuraciones de entorno y la evaluación visual del rendimiento del agente.
"""

import pygame
import numpy as np
import os
import time
import torch
from agent import Agent

# --- CONFIGURACIÓN GENERAL ---
"""Constantes de configuración para la interfaz gráfica."""
GRID_WIDTH = 5          # Ancho de la grilla en celdas
GRID_HEIGHT = 5         # Alto de la grilla en celdas
CELL_SIZE = 120         # Tamaño de cada celda en píxeles
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE    # Ancho total de la pantalla
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE  # Alto total de la pantalla

# Paleta de colores para la interfaz
COLOR_FONDO = (25, 25, 25)        # Fondo oscuro
COLOR_LINEAS = (40, 40, 40)       # Líneas de la grilla
COLOR_CURSOR = (255, 255, 0)      # Cursor amarillo en modo setup
COLOR_TEXTO = (230, 230, 230)     # Texto claro


# --- ENTORNO PARA DDQN (misma estructura visual que Q-learning) ---
class EntornoGrid:
    """
    Entorno de grilla con interfaz gráfica para agentes DDQN.
    
    Esta clase maneja tanto la lógica del entorno como su representación visual,
    proporcionando una interfaz interactiva para configurar y visualizar el
    comportamiento del agente. Compatible con la arquitectura DDQN.
    
    Attributes:
        size (int): Tamaño de la grilla
        agent_pos (tuple): Posición actual del agente (x, y)
        frutas (set): Conjunto de posiciones con frutas
        venenos (set): Conjunto de posiciones con venenos
        paredes (set): Conjunto de posiciones con paredes (obstáculos)
    """
    
    def __init__(self):
        """
        Inicializa el entorno de grilla con configuración por defecto.
        
        El agente comienza en la posición (0,0) y todos los conjuntos de
        elementos están vacíos inicialmente.
        """
        self.size = GRID_WIDTH
        self.agent_pos = (0, 0)
        self.frutas = set()
        self.venenos = set()
        self.paredes = set()

    def reset_a_configuracion_inicial(self):
        """
        Resetea el agente a su posición inicial sin modificar el entorno.
        
        Esta función es útil para reiniciar episodios manteniendo la misma
        configuración de frutas, venenos y paredes establecida en modo setup.
        
        Returns:
            np.array: Estado inicial del entorno después del reset.
        """
        self.agent_pos = (0, 0)
        return self.get_state()

    def limpiar_entorno(self):
        """
        Elimina todos los elementos del entorno excepto el agente.
        
        Esta función es útil para limpiar completamente el entorno y comenzar
        una nueva configuración desde cero en modo setup.
        """
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        """
        Ejecuta una acción del agente y actualiza el estado del entorno.
        
        Este método implementa la lógica de movimiento y las reglas del juego,
        incluyendo colisiones con paredes, recolección de frutas y penalizaciones
        por venenos.
        
        Args:
            accion (int): Acción a ejecutar
                - 0: Mover hacia arriba (y-1)
                - 1: Mover hacia abajo (y+1)
                - 2: Mover hacia la izquierda (x-1)
                - 3: Mover hacia la derecha (x+1)
        
        Returns:
            tuple: (estado, recompensa, terminado)
                - estado (np.array): Nuevo estado del entorno
                - recompensa (float): Recompensa obtenida por la acción
                - terminado (bool): True si el episodio ha terminado
        
        Lógica de recompensas:
            - Colisión con pared/límite: -0.1 (sin movimiento)
            - Movimiento válido: -0.05 (costo de vida)
            - Tocar veneno: -10.0 (penalización + reset a origen)
            - Recolectar fruta: +1.0
            - Completar nivel: +10.0 adicional
        """
        x, y = self.agent_pos
        
        # Calcular nueva posición según la acción
        if accion == 0:
            y -= 1    # Mover hacia arriba
        elif accion == 1:
            y += 1    # Mover hacia abajo
        elif accion == 2:
            x -= 1    # Mover hacia la izquierda
        elif accion == 3:
            x += 1    # Mover hacia la derecha

        # Verificar colisiones con límites de la grilla o paredes
        if (
            x < 0
            or x >= GRID_WIDTH
            or y < 0
            or y >= GRID_HEIGHT
            or (x, y) in self.paredes
        ):
            # Movimiento inválido: penalización menor y no se mueve
            return self.get_state(), -0.1, False

        # Movimiento válido: actualizar posición del agente
        self.agent_pos = (x, y)
        recompensa = -0.05  # Costo base por movimiento
        terminado = False

        # Verificar interacciones con elementos del entorno
        if self.agent_pos in self.venenos:
            # Penalización por tocar veneno y reset a posición inicial
            recompensa = -10.0
            self.agent_pos = (0, 0)
        elif self.agent_pos in self.frutas:
            # Recompensa por recolectar fruta
            recompensa = 1.0
            self.frutas.remove(self.agent_pos)
            
            # Verificar si se completó el nivel (no quedan frutas)
            if not self.frutas:
                recompensa += 10.0  # Bonus por completar
                terminado = True

        return self.get_state(), recompensa, terminado

    def get_state(self):
        """
        Obtiene la representación del estado actual como tensor 3D.
        
        Convierte el estado del entorno en un formato compatible con redes
        neuronales convolucionales, usando 3 canales para representar
        diferentes tipos de elementos.
        
        Returns:
            np.array: Tensor 3D de forma (3, size, size) donde:
                - Canal 0: Posición del agente (1.0 donde está el agente)
                - Canal 1: Posiciones de frutas (1.0 donde hay frutas)
                - Canal 2: Posiciones de venenos (1.0 donde hay venenos)
                
        Nota: Las paredes no se incluyen en el estado ya que son estáticas
              y se manejan a través de las restricciones de movimiento.
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
        Renderiza el entorno completo en la pantalla de Pygame.
        
        Este método se encarga de dibujar todos los elementos visuales del juego,
        incluyendo la grilla, elementos del entorno, el agente, el cursor (en modo setup)
        y la información de controles.
        
        Args:
            pantalla (pygame.Surface): Superficie donde dibujar
            modo_juego (str): Modo actual ("SETUP" o "PLAYING")
            cursor_pos (tuple): Posición del cursor en modo setup
            img_fruta (pygame.Surface): Imagen de la fruta
            img_veneno (pygame.Surface): Imagen del veneno
            img_pared (pygame.Surface): Imagen de la pared
            img_agente (pygame.Surface): Imagen del agente
        
        Elementos visuales renderizados:
            1. Fondo y grilla
            2. Paredes (obstáculos estáticos)
            3. Frutas (objetivos a recolectar)
            4. Venenos (elementos a evitar)
            5. Agente (jugador controlado por IA)
            6. Cursor (solo en modo setup)
            7. Información de controles y modo actual
        """
        # Limpiar pantalla con color de fondo
        pantalla.fill(COLOR_FONDO)
        
        # Dibujar líneas de la grilla (verticales)
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
            
        # Dibujar líneas de la grilla (horizontales)
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Dibujar elementos del entorno en orden de capas
        # 1. Paredes (fondo)
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0] * CELL_SIZE, pared[1] * CELL_SIZE))
            
        # 2. Frutas (objetivos)
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0] * CELL_SIZE, fruta[1] * CELL_SIZE))
            
        # 3. Venenos (peligros)
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0] * CELL_SIZE, veneno[1] * CELL_SIZE))

        # 4. Agente (primer plano)
        pantalla.blit(
            img_agente, (self.agent_pos[0] * CELL_SIZE, self.agent_pos[1] * CELL_SIZE)
        )

        # 5. Cursor (solo en modo setup)
        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        # Renderizar información textual
        font = pygame.font.Font(None, 24)
        
        # Información del modo actual
        texto_modo = font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO)
        
        # Controles para modo setup
        controles1 = font.render(
            "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar", True, COLOR_TEXTO
        )
        
        # Controles generales
        controles2 = font.render("P=Jugar, S=Setup", True, COLOR_TEXTO)
        
        # Posicionar textos en la parte inferior
        pantalla.blit(texto_modo, (10, SCREEN_HEIGHT + 5))
        pantalla.blit(controles1, (10, SCREEN_HEIGHT + 30))
        pantalla.blit(controles2, (10, SCREEN_HEIGHT + 55))


# --- MAIN CON INTERFAZ COMPLETA ---
def main():
    """
    Función principal que ejecuta la interfaz gráfica completa del sistema DDQN.
    
    Esta función implementa el bucle principal del programa, manejando:
    - Inicialización de Pygame y carga de recursos
    - Gestión de eventos de teclado para ambos modos
    - Alternancia entre modo setup y modo juego
    - Renderizado continuo de la interfaz
    - Ejecución automática del agente en modo juego
    
    Modos de operación:
        SETUP: Permite configurar el entorno interactivamente
            - Flechas: Mover cursor
            - F: Añadir/quitar fruta
            - V: Añadir/quitar veneno  
            - W: Añadir/quitar pared
            - C: Limpiar entorno
            
        PLAYING: El agente entrenado juega automáticamente
            - Usa el modelo DDQN cargado para tomar decisiones
            - Visualiza el comportamiento del agente en tiempo real
            - Termina automáticamente y vuelve a setup al completar
    """
    # Inicializar Pygame y crear ventana
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente DDQN - Come Frutas 🍓☠️")

    def cargar_img(nombre, color_fallback):
        """
        Carga una imagen desde archivo con fallback a color sólido.
        
        Esta función auxiliar intenta cargar una imagen desde el directorio
        del script. Si falla, crea una superficie de color sólido como respaldo.
        
        Args:
            nombre (str): Nombre del archivo de imagen
            color_fallback (tuple): Color RGB de respaldo si falla la carga
            
        Returns:
            pygame.Surface: Superficie escalada al tamaño de celda
        """
        try:
            ruta = os.path.join(os.path.dirname(__file__), nombre)
            img = pygame.image.load(ruta).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except:
            # Crear superficie de color sólido como fallback
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.fill(color_fallback)
            return surf

    # Cargar imágenes con colores de respaldo
    img_fruta = cargar_img("fruta.png", (0, 255, 0))      # Verde si falla
    img_veneno = cargar_img("veneno.png", (255, 0, 0))     # Rojo si falla
    img_pared = cargar_img("pared.png", (100, 100, 100))   # Gris si falla
    img_agente = cargar_img("agente.png", (0, 0, 255))     # Azul si falla

    # Inicializar componentes principales
    entorno = EntornoGrid()
    agente = Agent(state_shape=(3, GRID_HEIGHT, GRID_WIDTH), action_size=4)
    agente.load("dqn_model.pth")  # Cargar modelo entrenado

    # Variables de control de la interfaz
    cursor_pos = [0, 0]      # Posición del cursor en modo setup
    modo_juego = "SETUP"     # Modo inicial
    reloj = pygame.time.Clock()  # Control de FPS
    corriendo = True

    # Bucle principal del programa
    while corriendo:
        # Procesar eventos de Pygame
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                # Cambios de modo (disponibles en cualquier momento)
                if evento.key == pygame.K_p:
                    print("--- MODO JUEGO ---")
                    entorno.reset_a_configuracion_inicial()
                    modo_juego = "PLAYING"
                    time.sleep(0.5)  # Pausa para evitar acciones inmediatas

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                # Controles específicos del modo SETUP
                if modo_juego == "SETUP":
                    # Movimiento del cursor
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_HEIGHT - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_WIDTH - 1, cursor_pos[0] + 1)

                    # Gestión de elementos en la posición del cursor
                    pos = tuple(cursor_pos)
                    
                    if evento.key == pygame.K_f:
                        # Alternar fruta en posición actual
                        if pos in entorno.frutas:
                            entorno.frutas.remove(pos)
                        else:
                            entorno.frutas.add(pos)
                            # Remover otros elementos de la misma posición
                            entorno.venenos.discard(pos)
                            entorno.paredes.discard(pos)
                            
                    elif evento.key == pygame.K_v:
                        # Alternar veneno en posición actual
                        if pos in entorno.venenos:
                            entorno.venenos.remove(pos)
                        else:
                            entorno.venenos.add(pos)
                            # Remover otros elementos de la misma posición
                            entorno.frutas.discard(pos)
                            entorno.paredes.discard(pos)
                            
                    elif evento.key == pygame.K_w:
                        # Alternar pared en posición actual
                        if pos in entorno.paredes:
                            entorno.paredes.remove(pos)
                        else:
                            entorno.paredes.add(pos)
                            # Remover otros elementos de la misma posición
                            entorno.frutas.discard(pos)
                            entorno.venenos.discard(pos)
                            
                    elif evento.key == pygame.K_c:
                        # Limpiar todo el entorno
                        print("--- LIMPIANDO ENTORNO ---")
                        entorno.limpiar_entorno()

        # Lógica del modo PLAYING (agente automático)
        if modo_juego == "PLAYING":
            # Obtener estado actual del entorno
            estado = entorno.get_state()
            
            # El agente elige una acción usando el modelo entrenado
            # explore=False significa que usa solo explotación (sin exploración)
            accion = agente.choose_action(estado, explore=False)
            
            # Ejecutar la acción en el entorno
            _, _, terminado = entorno.step(accion)
            
            # Si el episodio terminó, volver al modo setup
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
                
            # Controlar velocidad de visualización (10 FPS para el agente)
            time.sleep(0.1)

        # Renderizado de la interfaz
        # Crear superficie completa incluyendo espacio para texto
        pantalla_con_info = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
        pantalla_con_info.fill(COLOR_FONDO)
        
        # Dibujar el entorno en la superficie
        entorno.dibujar(
            pantalla_con_info,
            modo_juego,
            tuple(cursor_pos),
            img_fruta,
            img_veneno,
            img_pared,
            img_agente,
        )
        
        # Copiar la superficie completa a la pantalla principal
        pantalla.blit(pantalla_con_info, (0, 0))
        
        # Actualizar la pantalla y controlar FPS
        pygame.display.flip()
        reloj.tick(60)  # Limitar a 60 FPS para suavidad visual

    # Limpiar recursos al salir
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa.
    
    Ejecuta la función main() solo cuando el archivo se ejecuta directamente,
    no cuando se importa como módulo. Esto permite reutilizar las clases
    y funciones en otros scripts sin ejecutar automáticamente la interfaz.
    """
    main()
