# environment.py
import numpy as np

class GridEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self, agent_pos=(0, 0), fruit_pos=[], poison_pos=[]):
        self.agent_pos = np.array(agent_pos)
        self.fruit_pos = [np.array(p) for p in fruit_pos]
        self.poison_pos = [np.array(p) for p in poison_pos]
        return self.get_state()

    def get_state(self):
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        for fruit in self.fruit_pos:
            state[1, fruit[0], fruit[1]] = 1.0
        for poison in self.poison_pos:
            state[2, poison[0], poison[1]] = 1.0
        return state

    def step(self, action):
        if action == 0: self.agent_pos[0] -= 1
        elif action == 1: self.agent_pos[0] += 1
        elif action == 2: self.agent_pos[1] -= 1
        elif action == 3: self.agent_pos[1] += 1
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        done = False
        reward = -0.1 # No se usa en el entrenamiento por imitación, pero es bueno tenerlo

        for i, fruit in enumerate(self.fruit_pos):
            if np.array_equal(self.agent_pos, fruit):
                self.fruit_pos.pop(i)
                break
        
        if any(np.array_equal(self.agent_pos, p) for p in self.poison_pos):
            done = True

        if not self.fruit_pos:
            done = True
        
        return self.get_state(), reward, done