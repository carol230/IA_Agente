# environment.py
import numpy as np

class GridEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.start_pos = (0, 0) # Guardaremos la posición inicial
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        """ Reinicia el entorno con una configuración específica. """
        self.start_pos = np.array(agent_pos) # Guardamos la posición inicial del episodio
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        # ... (esta función no cambia) ...
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        for fruit in self.fruit_pos:
            state[1, fruit[0], fruit[1]] = 1.0
        for poison in self.poison_pos:
            state[2, poison[0], poison[1]] = 1.0
        return state

    def step(self, action):
        # ... (la parte de mover el agente no cambia) ...
        if action == 0: self.agent_pos[0] -= 1
        elif action == 1: self.agent_pos[0] += 1
        elif action == 2: self.agent_pos[1] -= 1
        elif action == 3: self.agent_pos[1] += 1
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        reward = -0.05
        done = False

        # --- LÓGICA DE VENENO MODIFICADA ---
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            reward = -10.0
            self.agent_pos = np.copy(self.start_pos) # Lo devolvemos al inicio
            # IMPORTANTE: done NO es True. El episodio continúa.
        else:
            # El resto de la lógica solo se ejecuta si NO tocó un veneno
            eaten_fruit_this_step = False
            for i, fruit in enumerate(self.fruit_pos):
                if np.array_equal(self.agent_pos, fruit):
                    reward += 1.0
                    self.fruit_pos.pop(i)
                    eaten_fruit_this_step = True
                    break

            if not eaten_fruit_this_step and self.fruit_pos:
                # El reward shaping se puede mantener o quitar, en este punto es opcional.
                # Dejémoslo para que tenga una guía.
                pass # Puedes añadir la lógica de distancia si lo deseas

            if not self.fruit_pos:
                done = True
                reward += 10.0

        return self.get_state(), reward, done