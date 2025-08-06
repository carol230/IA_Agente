# dqn_agente_comefrutas.py
"""
Interfaz gráfica de demostración para agente DQN (Deep Q-Network).

Este módulo proporciona una interfaz visual interactiva para demostrar el
comportamiento de un agente DQN entrenado en el problema de recolección de frutas.
A diferencia del DDQN, esta implementación utiliza DQN clásico con una sola red.

Características principales:
- Interfaz de configuración interactiva para crear escenarios personalizados
- Visualización en tiempo real del comportamiento del agente entrenado
- Sistema de dos modos: configuración (SETUP) y ejecución (PLAYING)
- Compatibilidad con modelos DQN preentrenados
- Interfaz de usuario intuitiva con controles de teclado y mouse

El sistema está diseñado para:
- Demostraciones educativas del comportamiento de IA
- Validación visual del rendimiento del agente
- Experimentación rápida con diferentes configuraciones de entorno
- Evaluación cualitativa de estrategias aprendidas

Diferencias con DDQN:
- Utiliza una sola red neuronal (no red objetivo separada)
- Implementación más simple del algoritmo Q-learning
- Compatible con modelos entrenados usando DQN clásico
"""

import pygame
import numpy as np
import os
import time
from agent import Agent

# --- CONFIGURACIÓN DE LA INTERFAZ VISUAL ---
"""Parámetros de configuración para la ventana y visualización."""
GRID_WIDTH = 5          # Ancho de la grilla en número de celdas
GRID_HEIGHT = 5         # Alto de la grilla en número de celdas
CELL_SIZE = 120         # Tamaño de cada celda en píxeles (120x120)
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE    # Ancho total de la ventana (600px)
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE  # Alto total de la ventana (600px)

# --- ESQUEMA DE COLORES ---
"""Paleta de colores para una interfaz moderna y legible."""
COLOR_FONDO = (25, 25, 25)        # Fondo oscuro para reducir fatiga visual
COLOR_LINEAS = (40, 40, 40)       # Líneas de grilla sutiles
COLOR_CURSOR = (255, 255, 0)      # Cursor amarillo brillante para visibilidad
COLOR_TEXTO = (230, 230, 230)     # Texto claro sobre fondo oscuro


class EntornoGrid:
    """
    Entorno de grilla especializado para demostración de agentes DQN.
    
    Esta clase maneja tanto la lógica del entorno como su representación visual,
    proporcionando una plataforma completa para demostrar el comportamiento de
    agentes DQN entrenados en problemas de navegación y recolección.
    
    Características del entorno:
    - Grilla bidimensional con elementos configurables
    - Gestión de colisiones y límites
    - Sistema de recompensas integrado
    - Representación visual con Pygame
    - Compatibilidad con formato de estado DQN (tensor 3D)
    
    Elementos del entorno:
    - Agente: Entidad controlada por IA que debe recolectar frutas
    - Frutas: Objetivos que otorgan recompensas positivas
    - Venenos: Obstáculos que causan penalizaciones y reseteo
    - Paredes: Barreras físicas que bloquean el movimiento
    
    Attributes:
        size (int): Tamaño de la grilla (siempre cuadrada)
        agent_pos (tuple): Posición actual del agente (fila, columna)
        frutas (set): Conjunto de posiciones que contienen frutas
        venenos (set): Conjunto de posiciones que contienen venenos
        paredes (set): Conjunto de posiciones que contienen paredes
    """
    
    def __init__(self):
        """
        Inicializa el entorno con configuración por defecto.
        
        El entorno comienza con:
        - Agente en posición (0,0) - esquina superior izquierda
        - Todos los conjuntos de elementos vacíos
        - Tamaño de grilla determinado por GRID_WIDTH
        """
        self.size = GRID_WIDTH
        self.agent_pos = (0, 0)  # Posición inicial estándar
        self.frutas = set()      # Conjunto vacío inicialmente
        self.venenos = set()     # Conjunto vacío inicialmente
        self.paredes = set()     # Conjunto vacío inicialmente

    def reset_a_configuracion_inicial(self):
        """
        Reinicia solo la posición del agente sin modificar el entorno.
        
        Esta función es útil para comenzar nuevos episodios manteniendo
        la misma configuración de elementos (frutas, venenos, paredes)
        establecida durante el modo setup.
        
        Returns:
            np.array: Estado inicial del entorno después del reset
        """
        self.agent_pos = (0, 0)
        return self.get_state()

    def limpiar_entorno(self):
        """
        Elimina todos los elementos del entorno excepto el agente.
        
        Función de utilidad para limpiar completamente el entorno
        y comenzar una nueva configuración desde cero. Útil en
        modo setup para crear nuevos escenarios rápidamente.
        """
        self.frutas.clear()
        self.venenos.clear()
        self.paredes.clear()

    def step(self, accion):
        """
        Ejecuta una acción del agente y actualiza el estado del entorno.
        
        Este método implementa la lógica principal del entorno, incluyendo:
        - Procesamiento de movimientos del agente
        - Detección de colisiones con paredes y límites
        - Cálculo de recompensas según las interacciones
        - Gestión de condiciones de terminación
        - Manejo especial de venenos (penalización + reset)
        
        Args:
            accion (int): Acción a ejecutar por el agente
                0: Mover hacia arriba (fila-1)
                1: Mover hacia abajo (fila+1)
                2: Mover hacia la izquierda (columna-1)
                3: Mover hacia la derecha (columna+1)
        
        Returns:
            tuple: (nuevo_estado, recompensa, episodio_terminado)
                - nuevo_estado (np.array): Estado resultante
                - recompensa (float): Recompensa obtenida
                - episodio_terminado (bool): True si completó o falló
        
        Sistema de recompensas:
            - Colisión con pared/límite: -0.1 (movimiento inválido)
            - Movimiento válido: -0.05 (costo de vida)
            - Tocar veneno: -10.0 (penalización severa + reset a origen)
            - Recolectar fruta: +1.0 (recompensa por objetivo)
            - Completar nivel: +10.0 adicional (todas las frutas recolectadas)
        """
        # Obtener posición actual del agente
        fila, col = self.agent_pos
        
        # Calcular nueva posición según la acción
        if accion == 0:      # Arriba
            fila -= 1
        elif accion == 1:    # Abajo
            fila += 1
        elif accion == 2:    # Izquierda
            col -= 1
        elif accion == 3:    # Derecha
            col += 1

        # Verificar colisiones con límites de grilla o paredes
        if (
            fila < 0                        # Límite superior
            or fila >= GRID_HEIGHT          # Límite inferior
            or col < 0                      # Límite izquierdo
            or col >= GRID_WIDTH            # Límite derecho
            or (fila, col) in self.paredes  # Colisión con pared
        ):
            # Movimiento inválido: penalización menor, mantener posición
            return self.get_state(), -0.1, False
        
        # Movimiento válido: actualizar posición
        x, y = fila, col
        self.agent_pos = (x, y)
        recompensa = -0.05  # Costo base por movimiento (fomenta eficiencia)
        terminado = False

        # Procesar interacciones con elementos del entorno
        if self.agent_pos in self.venenos:
            # Penalización por tocar veneno y reset a posición inicial
            recompensa = -10.0
            self.agent_pos = (0, 0)  # Reset automático a origen
            
        elif self.agent_pos in self.frutas:
            # Recompensa por recolectar fruta
            recompensa = 1.0
            self.frutas.remove(self.agent_pos)  # Eliminar fruta recolectada
            
            # Verificar si se completó el nivel
            if not self.frutas:  # No quedan frutas
                recompensa += 10.0   # Bonus por completar
                terminado = True     # Episodio exitoso
                self.agent_pos = (0, 0)  # Reset para próximo episodio

        return self.get_state(), recompensa, terminado

    def get_state(self):
        """
        Convierte el estado actual del entorno a formato tensor para DQN.
        
        Esta función es crucial para la compatibilidad con redes neuronales
        convolucionales, transformando la representación discreta del entorno
        en un tensor 3D que puede ser procesado eficientemente por la CNN.
        
        Returns:
            np.array: Tensor 3D con forma (3, size, size) donde:
                - Canal 0: Posición del agente (1.0 donde está, 0.0 resto)
                - Canal 1: Posiciones de frutas (1.0 donde hay frutas)
                - Canal 2: Posiciones de venenos (1.0 donde hay venenos)
        
        Características del formato:
        - Tipo float32 para compatibilidad con PyTorch
        - Representación binaria (0.0 o 1.0) para claridad
        - Canales separados permiten que la CNN detecte patrones específicos
        - Dimensiones compatibles con arquitectura Conv2D
        
        Nota: Las paredes no se incluyen en el estado ya que son estáticas
              y el agente las aprende a través de las restricciones de movimiento.
        """
        # Inicializar tensor de estado con ceros
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
        Renderiza el entorno completo en la pantalla usando Pygame.
        
        Esta función maneja toda la visualización del entorno, incluyendo
        elementos del juego, interfaz de usuario y información contextual.
        
        Args:
            pantalla (pygame.Surface): Superficie donde renderizar
            modo_juego (str): Modo actual ("SETUP" o "PLAYING")
            cursor_pos (tuple): Posición del cursor en modo setup
            img_fruta (pygame.Surface): Imagen para representar frutas
            img_veneno (pygame.Surface): Imagen para representar venenos
            img_pared (pygame.Surface): Imagen para representar paredes
            img_agente (pygame.Surface): Imagen para representar al agente
        
        Proceso de renderizado:
        1. Limpiar pantalla con color de fondo
        2. Dibujar grilla de referencia
        3. Renderizar elementos por capas (paredes → frutas → venenos → agente)
        4. Mostrar cursor en modo setup
        5. Renderizar información de controles y estado
        
        El orden de renderizado es importante para la superposición correcta
        de elementos visuales y la legibilidad de la interfaz.
        """
        # Limpiar pantalla con color de fondo
        pantalla.fill(COLOR_FONDO)
        
        # Dibujar grilla de referencia
        # Líneas verticales
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (x, 0), (x, SCREEN_HEIGHT))
        # Líneas horizontales
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(pantalla, COLOR_LINEAS, (0, y), (SCREEN_WIDTH, y))

        # Renderizar elementos del entorno (orden de capas importante)
        # 1. Paredes (fondo) - obstáculos estáticos
        for pared in self.paredes:
            pantalla.blit(img_pared, (pared[0] * CELL_SIZE, pared[1] * CELL_SIZE))
            
        # 2. Frutas (objetivos) - elementos a recolectar
        for fruta in self.frutas:
            pantalla.blit(img_fruta, (fruta[0] * CELL_SIZE, fruta[1] * CELL_SIZE))
            
        # 3. Venenos (peligros) - elementos a evitar
        for veneno in self.venenos:
            pantalla.blit(img_veneno, (veneno[0] * CELL_SIZE, veneno[1] * CELL_SIZE))

        # 4. Agente (primer plano) - jugador controlado por IA
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

        # Renderizar información textual de la interfaz
        font = pygame.font.Font(None, 24)
        
        # Mostrar modo actual
        texto_modo = font.render(f"Modo: {modo_juego}", True, COLOR_TEXTO)
        
        # Instrucciones para modo setup
        controles1 = font.render(
            "SETUP: Flechas, F=Fruta, V=Veneno, W=Pared, C=Limpiar", True, COLOR_TEXTO
        )
        
        # Controles generales
        controles2 = font.render("P=Jugar, S=Setup", True, COLOR_TEXTO)
        
        # Posicionar textos en la parte inferior de la pantalla
        pantalla.blit(texto_modo, (10, SCREEN_HEIGHT + 5))
        pantalla.blit(controles1, (10, SCREEN_HEIGHT + 30))
        pantalla.blit(controles2, (10, SCREEN_HEIGHT + 55))


def main():
    """
    Función principal que ejecuta la interfaz de demostración DQN.
    
    Esta función implementa un sistema completo de demostración interactiva
    que permite a los usuarios configurar entornos personalizados y observar
    el comportamiento de un agente DQN entrenado.
    
    Flujo de la aplicación:
    1. Inicialización de Pygame y carga de recursos visuales
    2. Carga del agente DQN preentrenado
    3. Bucle principal con dos modos de operación:
       - SETUP: Configuración interactiva del entorno
       - PLAYING: Demostración del agente entrenado
    4. Renderizado continuo y gestión de eventos
    
    Modos de operación:
    
    **MODO SETUP (Configuración):**
    - Navegación con flechas del teclado
    - F: Añadir/quitar frutas en posición del cursor
    - V: Añadir/quitar venenos en posición del cursor
    - W: Añadir/quitar paredes en posición del cursor
    - C: Limpiar completamente el entorno
    
    **MODO PLAYING (Demostración):**
    - El agente DQN toma control automático
    - Visualización en tiempo real de decisiones
    - Finalización automática y retorno a setup
    
    **Controles Globales:**
    - P: Cambiar a modo PLAYING
    - S: Cambiar a modo SETUP
    - ESC/X: Salir de la aplicación
    """
    # Inicializar sistema gráfico Pygame
    pygame.init()
    pantalla = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
    pygame.display.set_caption("Agente DQN - Come Frutas 🍓")

    def cargar_img(nombre, color_fallback):
        """
        Función auxiliar para cargar imágenes con respaldo de color.
        
        Intenta cargar una imagen desde archivo y, si falla, crea una
        superficie de color sólido como alternativa. Esto garantiza que
        la aplicación funcione incluso sin los archivos de imagen.
        
        Args:
            nombre (str): Nombre/ruta del archivo de imagen
            color_fallback (tuple): Color RGB de respaldo (r, g, b)
            
        Returns:
            pygame.Surface: Superficie escalada al tamaño de celda
        """
        try:
            ruta = os.path.join(os.path.dirname(__file__), nombre)
            img = pygame.image.load(ruta).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        except:
            # Crear superficie de color sólido como respaldo
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.fill(color_fallback)
            return surf

    # Cargar recursos visuales con colores de respaldo
    img_fruta = cargar_img("../fruta.png", (0, 255, 0))      # Verde si falla
    img_veneno = cargar_img("../veneno.png", (255, 0, 0))     # Rojo si falla
    img_pared = cargar_img("../pared.png", (100, 100, 100))   # Gris si falla
    img_agente = cargar_img("../agente.png", (0, 0, 255))     # Azul si falla

    # Inicializar componentes principales
    entorno = EntornoGrid()                                    # Entorno de simulación
    agente = Agent(state_shape=(3, GRID_HEIGHT, GRID_WIDTH), action_size=4)  # Agente DQN
    agente.load("DQN/dqn_model.pth")                          # Cargar modelo preentrenado

    # Variables de control de la interfaz
    cursor_pos = [0, 0]        # Posición del cursor en modo setup
    modo_juego = "SETUP"       # Modo inicial
    reloj = pygame.time.Clock()  # Control de framerate
    corriendo = True           # Flag principal del bucle

    # Bucle principal de la aplicación
    while corriendo:
        # Procesar eventos del usuario
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN:
                # --- CONTROLES GLOBALES ---
                if evento.key == pygame.K_p:
                    print("--- MODO JUEGO ---")
                    entorno.reset_a_configuracion_inicial()
                    modo_juego = "PLAYING"
                    time.sleep(0.5)  # Pausa para evitar acciones inmediatas

                elif evento.key == pygame.K_s:
                    print("--- MODO SETUP ---")
                    modo_juego = "SETUP"

                # --- CONTROLES ESPECÍFICOS DEL MODO SETUP ---
                if modo_juego == "SETUP":
                    # Navegación del cursor con flechas del teclado
                    if evento.key == pygame.K_UP:
                        cursor_pos[1] = max(0, cursor_pos[1] - 1)
                    elif evento.key == pygame.K_DOWN:
                        cursor_pos[1] = min(GRID_HEIGHT - 1, cursor_pos[1] + 1)
                    elif evento.key == pygame.K_LEFT:
                        cursor_pos[0] = max(0, cursor_pos[0] - 1)
                    elif evento.key == pygame.K_RIGHT:
                        cursor_pos[0] = min(GRID_WIDTH - 1, cursor_pos[0] + 1)

                    # Obtener posición actual del cursor
                    pos = tuple(cursor_pos)
                    
                    # Gestión de elementos en la posición del cursor
                    if evento.key == pygame.K_f:
                        # Alternar fruta en posición actual
                        if pos in entorno.frutas:
                            entorno.frutas.remove(pos)
                        else:
                            entorno.frutas.add(pos)
                            # Limpiar otros elementos de la misma posición
                            entorno.venenos.discard(pos)
                            entorno.paredes.discard(pos)
                            
                    elif evento.key == pygame.K_v:
                        # Alternar veneno en posición actual
                        if pos in entorno.venenos:
                            entorno.venenos.remove(pos)
                        else:
                            entorno.venenos.add(pos)
                            # Limpiar otros elementos de la misma posición
                            entorno.frutas.discard(pos)
                            entorno.paredes.discard(pos)
                            
                    elif evento.key == pygame.K_w:
                        # Alternar pared en posición actual
                        if pos in entorno.paredes:
                            entorno.paredes.remove(pos)
                        else:
                            entorno.paredes.add(pos)
                            # Limpiar otros elementos de la misma posición
                            entorno.frutas.discard(pos)
                            entorno.venenos.discard(pos)
                            
                    elif evento.key == pygame.K_c:
                        # Limpiar completamente el entorno
                        print("--- LIMPIANDO ENTORNO ---")
                        entorno.limpiar_entorno()

        # --- LÓGICA DEL MODO PLAYING ---
        if modo_juego == "PLAYING":
            # Obtener estado actual del entorno
            estado = entorno.get_state()
            
            # El agente DQN elige la mejor acción (sin exploración)
            # explore=False garantiza que use solo la política aprendida
            accion = agente.choose_action(estado, explore=False)
            
            # Ejecutar la acción en el entorno
            _, _, terminado = entorno.step(accion)
            
            # Verificar si el episodio terminó
            if terminado:
                print("Juego terminado. Volviendo a SETUP.")
                modo_juego = "SETUP"
                
            # Control de velocidad para observación humana
            time.sleep(0.1)  # 10 FPS para visualización clara

        # --- SISTEMA DE RENDERIZADO ---
        # Crear superficie temporal para composición
        pantalla_con_info = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT + 80))
        pantalla_con_info.fill(COLOR_FONDO)
        
        # Renderizar el entorno completo en la superficie temporal
        entorno.dibujar(
            pantalla_con_info,
            modo_juego,
            tuple(cursor_pos),  # Convertir lista a tupla
            img_fruta,
            img_veneno,
            img_pared,
            img_agente,
        )
        
        # Transferir superficie temporal a pantalla principal
        pantalla.blit(pantalla_con_info, (0, 0))
        
        # Actualizar pantalla y controlar framerate
        pygame.display.flip()
        reloj.tick(60)  # Limitar a 60 FPS para suavidad

    # Limpiar recursos al salir de la aplicación
    pygame.quit()


if __name__ == "__main__":
    """
    Punto de entrada del programa de demostración DQN.
    
    Ejecuta la función main() cuando el archivo se ejecuta directamente.
    Este patrón permite importar clases y funciones de este módulo sin
    ejecutar automáticamente la interfaz de demostración.
    
    Uso típico:
        python dqn_agente_comefrutas.py  # Ejecuta la demostración
        
    La aplicación está diseñada para:
    - Demostraciones educativas de algoritmos DQN
    - Validación visual del comportamiento del agente
    - Experimentación rápida con configuraciones de entorno
    - Presentaciones de proyectos de IA/ML
    
    Diferencias con versión DDQN:
    - Utiliza algoritmo DQN clásico (una sola red)
    - Compatible con modelos entrenados con DQN simple
    - Interfaz idéntica pero agente subyacente diferente
    """
    main()
