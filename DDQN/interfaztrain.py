# ddqn_agente_comefrutas.py
"""
Interfaz de entrenamiento interactiva para agentes DDQN.

Este módulo proporciona una interfaz gráfica completa que permite:
- Configurar entornos de entrenamiento de manera interactiva
- Entrenar agentes DDQN con visualización en tiempo real
- Evaluar el rendimiento del agente después del entrenamiento
- Gestionar el ciclo completo de desarrollo de IA: setup → entrenamiento → evaluación

La interfaz integra tres modos principales:
1. SETUP: Configuración interactiva del entorno
2. TRAINING: Entrenamiento automático del agente DDQN
3. PLAYING: Evaluación visual del agente entrenado

Diseñado para facilitar la experimentación y el desarrollo iterativo de agentes
de aprendizaje por refuerzo en entornos de grilla.
"""

import pygame
import numpy as np
import os
import time
from agent import Agent
from environment import GridEnvironment

# --- CONFIGURACIÓN DEL ENTORNO Y VISUALIZACIÓN ---
"""Parámetros principales del sistema de entrenamiento."""
GRID_SIZE = 5           # Tamaño de la grilla (5x5)
CELL_SIZE = 120         # Tamaño en píxeles de cada celda
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE    # Ancho total de la ventana
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE   # Alto total de la ventana

# Esquema de colores para la interfaz
COLOR_FONDO = (25, 25, 25)        # Fondo oscuro para mejor contraste
COLOR_LINEAS = (40, 40, 40)       # Líneas sutiles de la grilla
COLOR_CURSOR = (255, 255, 0)      # Cursor amarillo brillante
COLOR_TEXTO = (230, 230, 230)     # Texto claro y legible

# --- PARÁMETROS DE ENTRENAMIENTO ---
"""Configuración del proceso de entrenamiento DDQN."""
NUM_EPISODIOS_ENTRENAMIENTO = 3000    # Número total de episodios de entrenamiento
BATCH_SIZE = 128                      # Tamaño del lote para replay de experiencias


def cargar_imagen(ruta, color_si_falla):
    """
    Carga una imagen desde archivo con sistema de fallback robusto.
    
    Esta función implementa un mecanismo de carga de imágenes que garantiza
    que el programa funcione incluso si los archivos de imagen no están
    disponibles, creando superficies de color como respaldo.
    
    Args:
        ruta (str): Ruta relativa o absoluta al archivo de imagen
        color_si_falla (tuple): Color RGB (r, g, b) a usar si falla la carga
        
    Returns:
        pygame.Surface: Superficie escalada al tamaño de celda, ya sea la
                       imagen cargada o una superficie de color sólido
                       
    Características:
        - Manejo automático de transparencia (convert_alpha)
        - Escalado automático al tamaño de celda
        - Fallback graceful a color sólido
        - Compatible con todos los formatos soportados por Pygame
    """
    try:
        # Intentar cargar la imagen desde archivo
        img = pygame.image.load(ruta).convert_alpha()
        # Escalar al tamaño exacto de celda para consistencia visual
        return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
    except pygame.error:
        # Si falla la carga, crear superficie de color sólido
        surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
        surf.fill(color_si_falla)
        return surf


def main():
    """
    Función principal que ejecuta la interfaz de entrenamiento DDQN.
    
    Esta función implementa un sistema completo de desarrollo de agentes IA que incluye:
    
    1. **Configuración Interactiva (Modo SETUP)**:
       - Diseño visual del entorno usando cursor
       - Colocación de frutas, venenos y paredes
       - Validación de configuraciones
    
    2. **Entrenamiento Automatizado (Modo TRAINING)**:
       - Ejecución de algoritmo DDQN completo
       - Actualización de redes objetivo
       - Monitoreo de progreso en tiempo real
       - Gestión de memoria de experiencias
    
    3. **Evaluación Visual (Modo PLAYING)**:
       - Visualización del comportamiento aprendido
       - Modo sin exploración (solo explotación)
       - Análisis cualitativo del rendimiento
    
    Flujo de trabajo típico:
        SETUP → TRAINING → PLAYING → [iteración]
    
    La interfaz permite experimentación rápida con diferentes configuraciones
    de entorno y hiperparámetros de entrenamiento.
    """
    # Inicialización del sistema gráfico
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente Come-Frutas DDQN 🍓☠️")

    # Cargar recursos visuales con colores de fallback específicos
    img_fruta = cargar_imagen("fruta.png", (40, 200, 40))    # Verde si falla
    img_veneno = cargar_imagen("veneno.png", (200, 40, 40))   # Rojo si falla
    img_pared = cargar_imagen("pared.png", (100, 100, 100))   # Gris si falla
    img_agente = cargar_imagen("agente.png", (40, 200, 40))   # Verde si falla

    # Inicialización de componentes principales del sistema
    entorno = GridEnvironment(size=GRID_SIZE)  # Entorno de simulación
    agente = Agent(state_shape=(3, GRID_SIZE, GRID_SIZE), action_size=4)  # Agente DDQN

    # Variables de control de la interfaz
    cursor_pos = [0, 0]        # Posición del cursor en modo setup
    modo_juego = "SETUP"       # Estado inicial del sistema
    reloj = pygame.time.Clock()  # Control de framerate
    corriendo = True           # Flag principal del bucle

    # Conjuntos para gestionar elementos del entorno configurables
    frutas = set()    # Posiciones de objetivos (recompensa positiva)
    venenos = set()   # Posiciones de peligros (penalización)
    paredes = set()   # Posiciones de obstáculos (bloqueo de movimiento)

    # Bucle principal del sistema de entrenamiento
    while corriendo:
        # Procesamiento de eventos del usuario
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                # --- CONTROL DE ENTRENAMIENTO (Tecla T) ---
                if evento.key == pygame.K_t:
                    if modo_juego != "TRAINING":
                        print("--- ENTRENANDO DDQN ---")
                        modo_juego = "TRAINING"
                        
                        # Bucle principal de entrenamiento DDQN
                        for episodio in range(NUM_EPISODIOS_ENTRENAMIENTO):
                            # Reiniciar entorno con configuración actual
                            estado = entorno.reset(
                                agent_pos=(0, 0),
                                fruit_pos=list(frutas),
                                poison_pos=list(venenos),
                            )
                            
                            # Variables de control del episodio
                            terminado = False
                            total_reward = 0
                            
                            # Bucle del episodio individual
                            while not terminado:
                                # El agente elige acción con exploración activa
                                accion = agente.choose_action(estado, explore=True)
                                
                                # Ejecutar acción y observar resultado
                                nuevo_estado, recompensa, terminado = entorno.step(accion)
                                
                                # Almacenar experiencia en memoria de replay
                                agente.remember(
                                    estado, accion, recompensa, nuevo_estado, terminado
                                )
                                
                                # Entrenar la red con experiencias pasadas
                                agente.replay(BATCH_SIZE)
                                
                                # Actualización periódica de la red objetivo
                                if agente.steps_done % agente.update_target_every == 0:
                                    agente.update_target_network()
                                
                                # Preparar para siguiente paso
                                estado = nuevo_estado
                                total_reward += recompensa
                            
                            # Reporte de progreso cada 100 episodios
                            if (episodio + 1) % 100 == 0:
                                print(
                                    f"Ep {episodio+1}, Reward: {total_reward:.2f}, Epsilon: {agente.epsilon:.3f}"
                                )
                        
                        print("--- ENTRENAMIENTO COMPLETO ---")
                        modo_juego = "PLAYING"  # Cambiar automáticamente a evaluación

                # --- CONTROL DE MODOS (Teclas P y S) ---
                elif evento.key == pygame.K_p:
                    print("--- MODO PLAYING ---")
                    # Reiniciar entorno para evaluación
                    entorno.reset(
                        agent_pos=(0, 0),
                        fruit_pos=list(frutas),
                        poison_pos=list(venenos),
                    )
                    modo_juego = "PLAYING"

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                # --- CONTROLES DEL MODO SETUP ---
                if modo_juego == "SETUP":
                    # Control de navegación del cursor
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_SIZE - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_SIZE - 1, cursor_pos[0] + 1)

                    # Conversión de coordenadas (cursor usa x,y pero entorno usa y,x)
                    pos = tuple(cursor_pos[::-1])
                    
                    # Gestión de elementos en la posición del cursor
                    if evento.key == pygame.K_f:
                        # Alternar fruta en posición actual
                        if pos in frutas:
                            frutas.remove(pos)
                        else:
                            frutas.add(pos)
                            # Limpiar otros elementos de la misma posición
                            venenos.discard(pos)
                            paredes.discard(pos)
                            
                    elif evento.key == pygame.K_v:
                        # Alternar veneno en posición actual
                        if pos in venenos:
                            venenos.remove(pos)
                        else:
                            venenos.add(pos)
                            # Limpiar otros elementos de la misma posición
                            frutas.discard(pos)
                            paredes.discard(pos)
                            
                    elif evento.key == pygame.K_w:
                        # Alternar pared en posición actual
                        if pos in paredes:
                            paredes.remove(pos)
                        else:
                            paredes.add(pos)
                            # Limpiar otros elementos de la misma posición
                            frutas.discard(pos)
                            venenos.discard(pos)
                            
                    elif evento.key == pygame.K_c:
                        # Limpiar completamente el entorno
                        frutas.clear()
                        venenos.clear()
                        paredes.clear()

        # --- LÓGICA DEL MODO PLAYING ---
        if modo_juego == "PLAYING":
            # Obtener estado actual del entorno
            estado = entorno.get_state()
            
            # El agente toma decisiones sin exploración (solo explotación)
            accion = agente.choose_action(estado, explore=False)
            
            # Ejecutar acción y verificar si terminó el episodio
            _, _, terminado = entorno.step(accion)
            
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
            
            # Control de velocidad de visualización
            time.sleep(0.1)  # 10 FPS para observar mejor el comportamiento

        # --- SISTEMA DE RENDERIZADO COMPLETO ---
        # Limpiar pantalla con color de fondo
        pantalla.fill(COLOR_FONDO)
        
        # Dibujar grilla de referencia
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Renderizar elementos del entorno (orden de capas importante)
        # Nota: Las coordenadas se invierten para coincidir con el sistema visual
        
        # 1. Paredes (capa de fondo)
        for pared in paredes:
            pantalla.blit(img_pared, (pared[1] * CELL_SIZE, pared[0] * CELL_SIZE))
            
        # 2. Frutas (objetivos)
        for fruta in frutas:
            pantalla.blit(img_fruta, (fruta[1] * CELL_SIZE, fruta[0] * CELL_SIZE))
            
        # 3. Venenos (peligros)
        for veneno in venenos:
            pantalla.blit(img_veneno, (veneno[1] * CELL_SIZE, veneno[0] * CELL_SIZE))

        # 4. Agente (solo visible fuera del modo setup)
        if modo_juego != "SETUP":
            pos = entorno.agent_pos
            pantalla.blit(img_agente, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE))

        # 5. Cursor (solo visible en modo setup)
        if modo_juego == "SETUP":
            cursor_rect = pygame.Rect(
                cursor_pos[0] * CELL_SIZE,
                cursor_pos[1] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(pantalla, COLOR_CURSOR, cursor_rect, 3)

        # --- INTERFAZ DE INFORMACIÓN Y CONTROLES ---
        font = pygame.font.Font(None, 24)
        
        # Mostrar modo actual
        pantalla.blit(
            font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO),
            (10, SCREEN_HEIGHT + 5),
        )
        
        # Controles para modo setup
        pantalla.blit(
            font.render(
                "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar",
                True,
                COLOR_TEXTO,
            ),
            (10, SCREEN_HEIGHT + 30),
        )
        
        # Controles generales del sistema
        pantalla.blit(
            font.render("T=Entrenar, P=Jugar, S=Setup", True, COLOR_TEXTO),
            (10, SCREEN_HEIGHT + 55),
        )

        # Actualizar pantalla y controlar framerate
        pygame.display.flip()
        reloj.tick(60)  # Limitar a 60 FPS para suavidad visual

    # Limpiar recursos al finalizar
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa de entrenamiento.
    
    Ejecuta la función main() cuando el archivo se ejecuta directamente.
    Este patrón permite importar las funciones y clases de este módulo
    en otros scripts sin ejecutar automáticamente la interfaz de entrenamiento.
    
    Uso típico:
        python interfaztrain.py  # Ejecuta la interfaz completa
        
    O desde otro script:
        from interfaztrain import cargar_imagen  # Solo importa funciones
    """
    main()
