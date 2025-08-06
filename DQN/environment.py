# environment.py
"""
Entorno de cuadrícula para el entrenamiento de un agente DQN.

Este módulo implementa un entorno de juego donde un agente debe navegar
por una cuadrícula para recoger frutas mientras evita venenos. El entorno
utiliza reward shaping para guiar al agente hacia las frutas.

"""

import numpy as np

class GridEnvironment:
    """
    Entorno de cuadrícula para un agente que debe recoger frutas y evitar venenos.
    
    El entorno consiste en una cuadrícula de tamaño configurable donde:
    - El agente se mueve en 4 direcciones (arriba, abajo, izquierda, derecha)
    - Las frutas proporcionan recompensas positivas
    - Los venenos proporcionan recompensas negativas y terminan el juego
    - El objetivo es recoger todas las frutas sin tocar venenos
    
    Attributes:
        size (int): Tamaño de la cuadrícula (size x size)
        agent_pos (np.array): Posición actual del agente [fila, columna]
        fruit_pos (list): Lista de posiciones de frutas
        poison_pos (list): Lista de posiciones de venenos
    """
    def __init__(self, size=5):
        """
        Inicializa el entorno de cuadrícula.
        
        Args:
            size (int, optional): Tamaño de la cuadrícula. Por defecto es 5x5.
        """
        self.size = size
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """
        Reinicia el entorno con una configuración específica.
        
        Establece las posiciones iniciales del agente, frutas y venenos.
        Si no se proporcionan posiciones, se usan listas vacías para frutas y venenos.
        
        Args:
            agent_pos (tuple, optional): Posición inicial del agente (fila, columna). 
                                       Por defecto (0, 0).
            fruit_pos (list, optional): Lista de tuplas con posiciones de frutas.
                                       Por defecto lista vacía.
            poison_pos (list, optional): Lista de tuplas con posiciones de venenos.
                                        Por defecto lista vacía.
        
        Returns:
            np.array: Estado inicial del entorno como array 3D (3, size, size).
        """
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        """
        Genera la representación del estado actual del entorno.
        
        El estado se representa como una "imagen" de 3 canales que puede ser
        procesada por una CNN. Cada canal representa un tipo de elemento:
        
        - Canal 0: Posición del agente (1.0 donde está el agente, 0.0 en el resto)
        - Canal 1: Posiciones de frutas (1.0 donde hay frutas, 0.0 en el resto)
        - Canal 2: Posiciones de venenos (1.0 donde hay venenos, 0.0 en el resto)
        
        Esta representación permite que el agente "vea" todo el entorno de una vez
        y facilita el procesamiento por redes neuronales convolucionales.
        
        Returns:
            np.array: Estado del entorno como array 3D de forma (3, size, size)
                     con valores float32.
        """
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Canal 0: Posición del agente
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Canal 1: Posiciones de las frutas
        for fruit in self.fruit_pos:
            state[1, fruit[0], fruit[1]] = 1.0
            
        # Canal 2: Posiciones de los venenos
        for poison in self.poison_pos:
            state[2, poison[0], poison[1]] = 1.0
            
        return state

    def step(self, action):
        """
        Ejecuta una acción en el entorno y retorna el resultado.
        
        Esta función implementa la lógica principal del juego, incluyendo:
        1. Movimiento del agente
        2. Cálculo de recompensas con reward shaping
        3. Detección de colisiones con frutas y venenos
        4. Determinación de condiciones de terminación
        
        El sistema de recompensas incluye:
        - Recompensa por acercarse a frutas (+0.1)
        - Castigo por alejarse de frutas (-0.15)
        - Recompensa por recoger frutas (+1.0)
        - Castigo por tocar veneno (-1.0, termina el juego)
        - Recompensa por completar el nivel (+5.0)
        - Castigo base por movimiento (-0.05, fomenta eficiencia)
        
        Args:
            action (int): Acción a realizar:
                         0 = Arriba (decrementar fila)
                         1 = Abajo (incrementar fila)
                         2 = Izquierda (decrementar columna)
                         3 = Derecha (incrementar columna)
        
        Returns:
            tuple: (nuevo_estado, recompensa, terminado)
                - nuevo_estado (np.array): Estado del entorno después de la acción
                - recompensa (float): Recompensa obtenida por la acción
                - terminado (bool): True si el episodio ha terminado
        """
        
        # FASE 1: REWARD SHAPING - Calcular distancia a fruta más cercana ANTES del movimiento
        # Esto permite dar recompensas por acercarse/alejarse de las frutas
        old_dist_to_fruit = float('inf')
        if self.fruit_pos:
            distances = [np.linalg.norm(self.agent_pos - fruit) for fruit in self.fruit_pos]
            old_dist_to_fruit = min(distances)

        
        # FASE 2: MOVIMIENTO DEL AGENTE
        # Actualizar la posición del agente basada en la acción seleccionada
        if action == 0:      # Arriba
            self.agent_pos[0] -= 1
        elif action == 1:    # Abajo
            self.agent_pos[0] += 1
        elif action == 2:    # Izquierda
            self.agent_pos[1] -= 1
        elif action == 3:    # Derecha
            self.agent_pos[1] += 1

        # Limitar la posición del agente a los límites del tablero
        # np.clip asegura que las coordenadas estén entre 0 y (size-1)
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        
        # FASE 3: CÁLCULO DE RECOMPENSAS
        
        # Recompensa base: pequeño castigo por cada movimiento para fomentar eficiencia
        reward = -0.05  
        done = False

        # REWARD SHAPING: Calcular nueva distancia y recompensar acercamiento a frutas
        # Esto ayuda al agente a aprender a navegar hacia las frutas incluso antes de alcanzarlas
        new_dist_to_fruit = float('inf')
        if self.fruit_pos:
            distances = [np.linalg.norm(self.agent_pos - fruit) for fruit in self.fruit_pos]
            new_dist_to_fruit = min(distances)

            # Recompensar por acercarse, castigar por alejarse
            if new_dist_to_fruit < old_dist_to_fruit:
                reward += 0.1   # Recompensa por acercarse a una fruta
            else:
                reward -= 0.15  # Castigo por alejarse (ligeramente mayor para evitar indecisión)

        
        # FASE 4: DETECCIÓN DE EVENTOS
        
        # Verificar si el agente recogió una fruta
        for i, fruit in enumerate(self.fruit_pos):
            if np.array_equal(self.agent_pos, fruit):
                reward += 1.0  # Gran recompensa por recoger fruta
                self.fruit_pos.pop(i)  # Remover la fruta del entorno
                break  # Solo puede recoger una fruta por paso
        
        # Verificar si el agente tocó veneno (termina el juego)
        if any(np.array_equal(self.agent_pos, poison) for poison in self.poison_pos):
            reward = -1.0  # Castigo severo y absoluto por tocar veneno
            done = True    # Terminar el episodio inmediatamente

        # Verificar condición de victoria: no quedan frutas
        if not self.fruit_pos:
            done = True
            reward += 5.0  # Gran recompensa bonus por completar el objetivo

        return self.get_state(), reward, done