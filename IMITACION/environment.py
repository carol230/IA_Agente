# environment.py
"""
Entorno de cuadrícula para aprendizaje por imitación de agentes.

Este módulo implementa un entorno de grid simplificado donde un agente
debe navegar para recolectar frutas mientras evita venenos. Está diseñado
específicamente para generar datos de demostración experta y entrenar
agentes mediante aprendizaje por imitación.

Clases:
    GridEnvironment: Entorno de cuadrícula con estados visuales 3D
"""
import numpy as np

class GridEnvironment:
    """
    Entorno de cuadrícula para simulación de navegación y recolección.
    
    Implementa un mundo de grid 2D donde el agente debe recolectar todas
    las frutas evitando venenos. El estado se representa como una imagen
    de 3 canales (agente, frutas, venenos) ideal para redes convolucionales.
    
    Características:
        - Grid cuadrado de tamaño configurable
        - Estados visuales como tensores 3D
        - Movimiento con límites del entorno
        - Detección automática de colisiones
        - Condiciones de terminación por victoria/derrota
    
    Attributes:
        size (int): Tamaño del grid (size x size)
        agent_pos (np.ndarray): Posición actual del agente [x, y]
        fruit_pos (list): Lista de posiciones de frutas [np.ndarray, ...]
        poison_pos (list): Lista de posiciones de venenos [np.ndarray, ...]
    """
    def __init__(self, size=5):
        """
        Inicializa el entorno de grid con tamaño especificado.
        
        Args:
            size (int): Dimensión del grid cuadrado (default: 5)
        """
        self.size = size
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """
        Reinicia el entorno con configuración específica de elementos.
        
        Establece posiciones iniciales del agente, frutas y venenos.
        Utilizado para crear escenarios específicos para generación
        de datos de demostración o evaluación de políticas.
        
        Args:
            agent_pos (tuple): Posición inicial del agente (x, y) (default: (0,0))
            fruit_pos (list): Lista de posiciones de frutas [(x,y), ...] (default: [])
            poison_pos (list): Lista de posiciones de venenos [(x,y), ...] (default: [])
        
        Returns:
            np.ndarray: Estado inicial del entorno con forma (3, size, size)
        
        Note:
            Las listas de posiciones se convierten a arrays numpy para
            operaciones vectorizadas eficientes durante la simulación.
        """
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        """
        Genera representación visual del estado actual como tensor 3D.
        
        Crea una imagen de 3 canales donde cada canal representa un tipo
        de elemento del entorno. Esta representación es ideal para redes
        convolucionales que procesan información espacial.
        
        Estructura de canales:
            - Canal 0: Posición del agente (binario)
            - Canal 1: Posiciones de frutas (binario)
            - Canal 2: Posiciones de venenos (binario)
        
        Returns:
            np.ndarray: Estado con forma (3, size, size) y dtype float32
                       Valores: 1.0 para presencia de elemento, 0.0 para ausencia
        
        Example:
            Para grid 3x3 con agente en (0,0) y fruta en (1,1):
            Canal 0: [[1, 0, 0],    Canal 1: [[0, 0, 0],    Canal 2: [[0, 0, 0],
                      [0, 0, 0],              [0, 1, 0],              [0, 0, 0],
                      [0, 0, 0]]              [0, 0, 0]]              [0, 0, 0]]
        """
        # Inicializar tensor de estado con ceros
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Canal 0: Posición del agente
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Canal 1: Posiciones de frutas
        for fruit in self.fruit_pos:
            state[1, fruit[0], fruit[1]] = 1.0
        
        # Canal 2: Posiciones de venenos
        for poison in self.poison_pos:
            state[2, poison[0], poison[1]] = 1.0
        
        return state

    def step(self, action):
        """
        Ejecuta una acción en el entorno y actualiza el estado.
        
        Procesa el movimiento del agente, maneja colisiones con límites,
        detecta recolección de frutas y verifica condiciones de terminación.
        Implementa la lógica core del entorno para simulación de episodios.
        
        Flujo de ejecución:
            1. Actualizar posición según acción
            2. Aplicar límites del entorno
            3. Procesar recolección de frutas
            4. Verificar colisiones con venenos
            5. Evaluar condiciones de terminación
        
        Args:
            action (int): Acción a ejecutar
                         0 = Arriba (decrementar x)
                         1 = Abajo (incrementar x)
                         2 = Izquierda (decrementar y)
                         3 = Derecha (incrementar y)
        
        Returns:
            tuple: (nuevo_estado, reward, done)
                - nuevo_estado (np.ndarray): Estado resultante (3, size, size)
                - reward (float): Recompensa por la acción (-0.1 por defecto)
                - done (bool): True si episodio terminó, False en caso contrario
        
        Note:
            El reward no se utiliza en aprendizaje por imitación pero se
            mantiene para compatibilidad con interfaces de RL estándar.
        """
        # Actualizar posición del agente según la acción
        if action == 0: 
            self.agent_pos[0] -= 1  # Arriba
        elif action == 1: 
            self.agent_pos[0] += 1  # Abajo
        elif action == 2: 
            self.agent_pos[1] -= 1  # Izquierda
        elif action == 3: 
            self.agent_pos[1] += 1  # Derecha
        
        # Aplicar límites del entorno (clipping)
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # Inicializar variables de terminación
        done = False
        reward = -0.1  # Penalización por paso (no usado en imitación)

        # Verificar recolección de frutas
        for i, fruit in enumerate(self.fruit_pos):
            if np.array_equal(self.agent_pos, fruit):
                # Fruta recolectada: eliminar de la lista
                self.fruit_pos.pop(i)
                break
        
        # Verificar colisión con venenos (derrota)
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            done = True

        # Verificar victoria (todas las frutas recolectadas)
        if not self.fruit_pos:
            done = True
        
        return self.get_state(), reward, done