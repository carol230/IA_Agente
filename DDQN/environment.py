# environment.py
"""
Entorno de grilla para el entrenamiento de agentes de aprendizaje por refuerzo.
Este módulo implementa un entorno de grilla donde un agente debe recolectar frutas 
mientras evita venenos, diseñado específicamente para algoritmos DDQN.
"""

import numpy as np

class GridEnvironment:
    """
    Entorno de grilla 2D para simulación de agentes que recolectan frutas y evitan venenos.
    
    El entorno consiste en una grilla cuadrada donde:
    - El agente se mueve en 4 direcciones (arriba, abajo, izquierda, derecha)
    - Las frutas otorgan recompensas positivas cuando son recolectadas
    - Los venenos causan penalizaciones y resetean la posición del agente
    - El objetivo es recolectar todas las frutas minimizando las penalizaciones
    
    Attributes:
        size (int): Tamaño de la grilla (size x size)
        start_pos (tuple): Posición inicial del agente en cada episodio
        agent_pos (np.array): Posición actual del agente
        fruit_pos (list): Lista de posiciones de las frutas
        poison_pos (list): Lista de posiciones de los venenos
    """
    
    def __init__(self, size=5):
        """
        Inicializa el entorno de grilla.
        
        Args:
            size (int, optional): Tamaño de la grilla cuadrada. Por defecto es 5x5.
        """
        self.size = size
        self.start_pos = (0, 0)  # Posición inicial por defecto
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """
        Reinicia el entorno con una configuración específica.
        
        Este método prepara el entorno para un nuevo episodio, estableciendo las posiciones
        iniciales del agente, frutas y venenos. Es crucial para el entrenamiento ya que
        permite configurar diferentes escenarios de aprendizaje.
        
        Args:
            agent_pos (tuple, optional): Posición inicial del agente (x, y). Por defecto (0, 0).
            fruit_pos (list, optional): Lista de tuplas con posiciones de frutas. Por defecto vacía.
            poison_pos (list, optional): Lista de tuplas con posiciones de venenos. Por defecto vacía.
        
        Returns:
            np.array: Estado inicial del entorno como tensor 3D (canales, altura, anchura).
        """
        self.start_pos = np.array(agent_pos)  # Guardamos la posición inicial del episodio
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        """
        Obtiene el estado actual del entorno como una representación tensorial.
        
        El estado se representa como un tensor 3D de forma (3, size, size) donde:
        - Canal 0: Posición del agente (1.0 en la posición actual, 0.0 en el resto)
        - Canal 1: Posiciones de las frutas (1.0 donde hay frutas, 0.0 en el resto)
        - Canal 2: Posiciones de los venenos (1.0 donde hay venenos, 0.0 en el resto)
        
        Esta representación permite que las redes neuronales procesen eficientemente
        la información espacial del entorno usando convoluciones.
        
        Returns:
            np.array: Tensor 3D de forma (3, size, size) representando el estado actual.
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
        Ejecuta una acción en el entorno y devuelve el resultado.
        
        Este método implementa la lógica principal del entorno, procesando las acciones
        del agente y calculando las recompensas correspondientes. Incluye manejo especial
        para venenos que resetean la posición del agente sin terminar el episodio.
        
        Args:
            action (int): Acción a ejecutar
                - 0: Mover hacia arriba (decrementar fila)
                - 1: Mover hacia abajo (incrementar fila)
                - 2: Mover hacia la izquierda (decrementar columna)
                - 3: Mover hacia la derecha (incrementar columna)
        
        Returns:
            tuple: (nuevo_estado, recompensa, episodio_terminado)
                - nuevo_estado (np.array): Estado resultante después de la acción
                - recompensa (float): Recompensa obtenida por la acción
                - episodio_terminado (bool): True si el episodio ha terminado
        
        Lógica de recompensas:
            - Movimiento básico: -0.05 (costo de vida)
            - Tocar veneno: -10.0 (penalización fuerte + reset a posición inicial)
            - Recolectar fruta: +1.0 (recompensa por objetivo)
            - Completar nivel: +10.0 (bonus por recolectar todas las frutas)
        """
        # Ejecutar el movimiento según la acción seleccionada
        if action == 0: 
            self.agent_pos[0] -= 1    # Mover hacia arriba
        elif action == 1: 
            self.agent_pos[0] += 1    # Mover hacia abajo
        elif action == 2: 
            self.agent_pos[1] -= 1    # Mover hacia la izquierda
        elif action == 3: 
            self.agent_pos[1] += 1    # Mover hacia la derecha
        
        # Asegurar que el agente permanezca dentro de los límites de la grilla
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # Recompensa base por cada movimiento (costo de vida)
        reward = -0.05
        done = False

        # --- LÓGICA DE MANEJO DE VENENOS ---
        # Verificar si el agente tocó algún veneno
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            reward = -10.0  # Penalización severa por tocar veneno
            self.agent_pos = np.copy(self.start_pos)  # Resetear a posición inicial
            # IMPORTANTE: done NO es True. El episodio continúa después del reset.
        else:
            # --- LÓGICA DE RECOLECCIÓN DE FRUTAS ---
            # Esta lógica solo se ejecuta si NO se tocó un veneno
            eaten_fruit_this_step = False
            
            # Verificar si el agente recolectó alguna fruta
            for i, fruit in enumerate(self.fruit_pos):
                if np.array_equal(self.agent_pos, fruit):
                    reward += 1.0  # Recompensa por recolectar fruta
                    self.fruit_pos.pop(i)  # Remover la fruta recolectada
                    eaten_fruit_this_step = True
                    break

            # Opcional: Aquí se puede agregar reward shaping basado en distancia
            if not eaten_fruit_this_step and self.fruit_pos:
                # Ejemplo: reward += -0.01 * distancia_a_fruta_más_cercana
                pass  # Actualmente no implementado

            # --- CONDICIÓN DE VICTORIA ---
            # Si no quedan frutas, el episodio termina exitosamente
            if not self.fruit_pos:
                done = True
                reward += 10.0  # Bonus por completar el nivel

        return self.get_state(), reward, done