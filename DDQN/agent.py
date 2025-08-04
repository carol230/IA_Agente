# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. Definir la Red Neuronal (el cerebro) ---
class CNN_DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN_DQN, self).__init__()
        # Capas convolucionales para "ver" patrones en el tablero
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Función para calcular el tamaño de salida de las capas conv
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        # Capas lineales para tomar decisiones basadas en lo que "vio"
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Aplanar la salida para las capas lineales
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)
    
# --- 2. Definir el Agente que usa la red ---

class Agent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        # --- CAMBIOS AQUÍ ---
        self.memory = deque(maxlen=50000)      # 1. Aumentar drásticamente la memoria
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001          # 2. Reducir la tasa de aprendizaje para un ajuste fino
        # --- FIN DE CAMBIOS ---
        self.update_target_every = 5

        h, w = state_shape[1], state_shape[2]
        self.model = CNN_DQN(h, w, action_size)
        self.target_model = CNN_DQN(h, w, action_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.steps_done = 0

    def update_target_network(self):
        """ Copia los pesos del modelo principal al modelo de destino. """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, explore=True):
        self.steps_done += 1 # Incrementar el contador de pasos
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size):
        """ Proceso de aprendizaje usando la lógica de Double DQN. """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1)

        # Predecir los Q-values para los estados actuales
        current_q_values = self.model(states).gather(1, actions)
        
        # --- LÓGICA DE DOUBLE DQN ---
        with torch.no_grad():
            # 1. Usar el modelo principal (self.model) para SELECCIONAR la mejor acción del siguiente estado.
            best_next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            
            # 2. Usar el modelo de destino (self.target_model) para EVALUAR el valor de esa acción seleccionada.
            next_q_values_target = self.target_model(next_states).gather(1, best_next_actions)
        # --- FIN DE LA LÓGICA ---

        # Calcular el valor de destino (target)
        target_q_values = rewards + (self.gamma * next_q_values_target * (~dones))
        
        # Calcular la pérdida
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Carga los pesos del modelo desde un archivo."""
        self.model.load_state_dict(torch.load(name))
        self.update_target_network() # Asegura que la red de destino también se actualice

    def save(self, name):
        """Guarda los pesos del modelo en un archivo."""
        torch.save(self.model.state_dict(), name)