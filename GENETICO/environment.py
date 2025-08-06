# environment.py
"""
Entorno de cuadrícula especializado para algoritmos genéticos.

Este módulo implementa un entorno modificado para el entrenamiento de agentes mediante
algoritmos genéticos. La principal diferencia con el entorno DQN es el manejo de venenos:
en lugar de terminar el episodio, el agente es enviado de vuelta a la posición inicial
con una penalización, permitiendo episodios más largos y mejor evaluación de fitness.

Características específicas para GA:
- Venenos no terminan el episodio, sino que resetean la posición
- Episodios más largos para mejor evaluación de fitness
- Recompensas ajustadas para discriminar mejor entre agentes
- Seguimiento de posición inicial para reset de venenos

"""

import numpy as np

class GridEnvironment:
    """
    Entorno de cuadrícula optimizado para algoritmos genéticos.
    
    Este entorno está diseñado específicamente para la evaluación de agentes
    mediante algoritmos genéticos. La principal modificación es que tocar venenos
    no termina el episodio, sino que envía al agente de vuelta al inicio,
    permitiendo episodios más largos y una mejor discriminación entre agentes.
    
    Características para GA:
    - Episodios más largos para mejor evaluación de fitness
    - Venenos causan reset de posición en lugar de game over
    - Recompensas ajustadas para mejor selección evolutiva
    - Seguimiento de posición inicial para mecánica de reset
    
    Attributes:
        size (int): Tamaño de la cuadrícula (size x size)
        start_pos (np.array): Posición inicial del agente en el episodio
        agent_pos (np.array): Posición actual del agente
        fruit_pos (list): Lista de posiciones de frutas
        poison_pos (list): Lista de posiciones de venenos
    """
    def __init__(self, size=5):
        """
        Inicializa el entorno de cuadrícula para algoritmos genéticos.
        
        Args:
            size (int, optional): Tamaño de la cuadrícula. Por defecto es 5x5.
        """
        self.size = size
        self.start_pos = (0, 0)  # Guardar posición inicial para reset de venenos
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """
        Reinicia el entorno con una configuración específica.
        
        Establece las posiciones iniciales y guarda la posición de inicio del agente
        para la mecánica de reset por venenos. Esta posición inicial es crucial
        en el paradigma de algoritmos genéticos ya que permite que el agente
        continúe intentando después de errores.
        
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
        self.start_pos = np.array(agent_pos)  # Guardar posición inicial del episodio
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        """
        Genera la representación del estado actual del entorno.
        
        Idéntica implementación al entorno DQN. El estado se representa como una 
        "imagen" de 3 canales que puede ser procesada por redes convolucionales.
        
        - Canal 0: Posición del agente (1.0 donde está el agente, 0.0 en el resto)
        - Canal 1: Posiciones de frutas (1.0 donde hay frutas, 0.0 en el resto)  
        - Canal 2: Posiciones de venenos (1.0 donde hay venenos, 0.0 en el resto)
        
        Esta representación permite que el agente "vea" todo el entorno de una vez
        y es compatible con arquitecturas de redes neuronales convolucionales.
        
        Returns:
            np.array: Estado del entorno como array 3D de forma (3, size, size)
                     con valores float32.
        """
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Canal 0: Posición del agente
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
        Ejecuta una acción en el entorno optimizado para algoritmos genéticos.
        
        Esta función implementa la lógica principal del juego con modificaciones
        específicas para algoritmos genéticos. La diferencia clave es el manejo
        de venenos: en lugar de terminar el episodio, el agente se resetea a la
        posición inicial, permitiendo episodios más largos y mejor evaluación.
        
        Diferencias con DQN:
        - Venenos NO terminan el episodio
        - Venenos resetean la posición del agente al inicio
        - Penalización mayor por venenos (-10.0 vs -1.0)
        - Episodios más largos para mejor discriminación de fitness
        
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
                - terminado (bool): True solo si todas las frutas fueron recogidas
        """
        
        # FASE 1: MOVIMIENTO DEL AGENTE
        # Lógica idéntica al entorno DQN
        if action == 0: 
            self.agent_pos[0] -= 1    # Arriba
        elif action == 1: 
            self.agent_pos[0] += 1    # Abajo
        elif action == 2: 
            self.agent_pos[1] -= 1    # Izquierda
        elif action == 3: 
            self.agent_pos[1] += 1    # Derecha
            
        # Limitar posición a los límites del tablero
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # FASE 2: INICIALIZACIÓN DE RECOMPENSAS
        reward = -0.05  # Pequeño castigo por cada movimiento
        done = False

        # FASE 2: INICIALIZACIÓN DE RECOMPENSAS
        reward = -0.05  # Pequeño castigo por cada movimiento
        done = False

        # FASE 3: MANEJO ESPECIAL DE VENENOS (DIFERENCIA CLAVE CON DQN)
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            # Veneno tocado: penalización severa pero NO termina el episodio
            reward = -10.0
            # CARACTERÍSTICA PRINCIPAL: Reset a posición inicial
            self.agent_pos = np.copy(self.start_pos)
            # CRÍTICO: done permanece False, el episodio continúa
            print("🔄 Agente tocó veneno, reseteado a posición inicial")
        else:
            # FASE 4: LÓGICA NORMAL (SOLO SI NO HAY VENENO)
            # Verificar si se recogió una fruta
            eaten_fruit_this_step = False
            for i, fruit in enumerate(self.fruit_pos):
                if np.array_equal(self.agent_pos, fruit):
                    reward += 1.0  # Recompensa por fruta
                    self.fruit_pos.pop(i)  # Remover fruta del entorno
                    eaten_fruit_this_step = True
                    print("🍎 Fruta recogida!")
                    break  # Solo una fruta por paso

            # Reward shaping opcional (sin implementar aquí)
            if not eaten_fruit_this_step and self.fruit_pos:
                # Aquí se podría agregar lógica de distancia como en DQN
                # Dejado como comentario para mantener simplicidad
                pass

            # FASE 5: CONDICIÓN DE VICTORIA
            if not self.fruit_pos:
                done = True
                reward += 10.0  # Gran recompensa por completar el nivel
                print("🏆 ¡Todas las frutas recogidas! Episodio completado")

        return self.get_state(), reward, done