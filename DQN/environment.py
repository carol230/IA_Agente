# environment.py
import numpy as np

class GridEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """ Reinicia el entorno con una configuración específica. """
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        """
        Representa el estado como una "imagen" de 3 canales (3x5x5).
        - Canal 0: Agente
        - Canal 1: Frutas
        - Canal 2: Venenos
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
        """ Realiza una acción y devuelve (nuevo_estado, recompensa, terminado) """
        
        # --- LÓGICA DE REWARD SHAPING (NUEVO) ---
        # 1. Calcular la distancia a la fruta más cercana ANTES de moverse.
        old_dist_to_fruit = float('inf')
        if self.fruit_pos:
            distances = [np.linalg.norm(self.agent_pos - fruit) for fruit in self.fruit_pos]
            old_dist_to_fruit = min(distances)
        # ----------------------------------------

        # Mover el agente (lógica original)
        if action == 0:
            self.agent_pos[0] -= 1
        elif action == 1:
            self.agent_pos[0] += 1
        elif action == 2:
            self.agent_pos[1] -= 1
        elif action == 3:
            self.agent_pos[1] += 1

        # Limitar al tablero (lógica original)
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # --- CÁLCULO DE RECOMPENSAS MEJORADO ---
        
        # Recompensa base por movimiento (un poco menos de castigo ahora)
        reward = -0.05  
        done = False

        # --- LÓGICA DE REWARD SHAPING (NUEVO) ---
        # 2. Calcular la nueva distancia y dar una recompensa/castigo por acercarse/alejarse.
        new_dist_to_fruit = float('inf')
        if self.fruit_pos:
            distances = [np.linalg.norm(self.agent_pos - fruit) for fruit in self.fruit_pos]
            new_dist_to_fruit = min(distances)

            if new_dist_to_fruit < old_dist_to_fruit:
                reward += 0.1  # Recompensa por acercarse a una fruta
            else:
                reward -= 0.15 # Castigo por alejarse (un poco más fuerte para evitar indecisión)
        # ----------------------------------------

        # Chequear si comió fruta (lógica original, pero con recompensa más alta)
        for i, fruit in enumerate(self.fruit_pos):
            if np.array_equal(self.agent_pos, fruit):
                reward += 1.0  # La recompensa por comer se suma a la recompensa de movimiento
                self.fruit_pos.pop(i)
                break
        
        # Chequear si tocó veneno (lógica original)
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            reward = -1.0 # El castigo por veneno es absoluto y termina el juego
            done = True 

        # Si no quedan frutas, el juego termina exitosamente (lógica original, recompensa más alta)
        if not self.fruit_pos:
            done = True
            reward += 5.0 # ¡Gran recompensa por completar el objetivo!

        return self.get_state(), reward, done