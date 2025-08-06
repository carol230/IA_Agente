# agent.py
"""
Implementación del agente DQN (Deep Q-Network) con arquitectura CNN.

Este módulo contiene la implementación completa del algoritmo DQN, incluyendo:
- Red neuronal convolucional para procesamiento de estados espaciales.
- Sistema de memoria de replay para entrenamiento estable.
- Estrategia epsilon-greedy para balancear exploración/explotación.
- Red objetivo para estabilizar el cálculo de los valores Q.
- Optimización con el optimizador Adam.

El agente está diseñado específicamente para problemas de navegación en grillas
donde el estado se representa como imágenes multi-canal, aprovechando las
capacidades de las CNNs para reconocer patrones espaciales.

Características principales:
- Arquitectura CNN optimizada para grillas pequeñas.
- Memoria de replay para descorrelacionar experiencias.
- Actualización periódica de la red objetivo.
- Técnicas de estabilización (gradient clipping, target network).
- Sistema de guardado/carga de modelos entrenados.

Referencias:
- DQN: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. RED NEURONAL CONVOLUCIONAL PARA DQN ---
class CNN_DQN(nn.Module):
    """
    Red neuronal convolucional optimizada para Q-learning en entornos de grilla.
    
    Esta arquitectura está diseñada para procesar estados representados como
    tensores 3D (canales x altura x anchura).
    
    Arquitectura:
    1. **Capas Convolucionales**: Para extraer características espaciales.
       - Conv1: 3->16 canales.
       - Conv2: 16->32 canales.
    
    2. **Capas Completamente Conectadas**: Para tomar decisiones basadas en las características.
       - FC1: 256 neuronas.
       - FC2: Salida de valores Q para cada acción.
    
    Args:
        h (int): Altura de la grilla de entrada.
        w (int): Anchura de la grilla de entrada.
        outputs (int): Número de acciones posibles.
    """
    
    def __init__(self, h, w, outputs):
        super(CNN_DQN, self).__init__()
        
        # --- CAPAS CONVOLUCIONALES ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # --- CÁLCULO DINÁMICO DEL TAMAÑO DE CARACTERÍSTICAS ---
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        # --- CAPAS COMPLETAMENTE CONECTADAS ---
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        """
        Propagación hacia adelante de la red.
        
        Procesa el estado de entrada a través de las capas para generar
        valores Q para cada acción posible.
        
        Args:
            x (torch.Tensor): Estado de entrada con forma (batch, 3, height, width).
        
        Returns:
            torch.Tensor: Valores Q para cada acción con forma (batch, num_actions).
        """
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)
    
    
# --- 2. AGENTE DQN CON MEMORIA DE REPLAY Y RED OBJETIVO ---
class Agent:
    """
    Agente de aprendizaje por refuerzo que implementa el algoritmo DQN.
    
    Este agente combina varias técnicas clave de deep reinforcement learning:
    
    **Componentes principales:**
    1. **Red Principal**: Se entrena activamente y decide las acciones.
    2. **Red Objetivo**: Una copia de la red principal que se actualiza lentamente,
       proporcionando targets estables para el entrenamiento y reduciendo oscilaciones.
    3. **Memoria de Replay**: Almacena experiencias para un aprendizaje más estable.
    4. **Estrategia Epsilon-Greedy**: Balancea entre explorar el entorno y explotar el conocimiento.
    
    Args:
        state_shape (tuple): Forma del estado (canales, altura, anchura).
        action_size (int): Número de acciones posibles en el entorno.
    """
    
    def __init__(self, state_shape, action_size):
        # --- CONFIGURACIÓN BÁSICA ---
        self.state_shape = state_shape
        self.action_size = action_size
        
        # --- MEMORIA DE REPLAY ---
        # Almacena tuplas de (estado, acción, recompensa, siguiente_estado, terminado).
        self.memory = deque(maxlen=20000)
        
        # --- HIPERPARÁMETROS DE APRENDIZAJE ---
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.update_target_every = 5
        
        # --- INICIALIZACIÓN DE REDES NEURONALES ---
        h, w = state_shape[1], state_shape[2]
        self.model = CNN_DQN(h, w, action_size)
        self.target_model = CNN_DQN(h, w, action_size)
        self.update_target_network()
        
        # --- CONFIGURACIÓN DE OPTIMIZACIÓN ---
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.steps_done = 0

    def update_target_network(self):
        """
        Actualiza la red objetivo copiando los pesos de la red principal.
        
        Esta operación es fundamental en DQN para mantener los targets estables
        durante el entrenamiento, evitando que el objetivo cambie en cada paso.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en la memoria de replay.
        
        Esto permite al agente aprender de un conjunto de experiencias pasadas y
        no correlacionadas, lo que estabiliza el entrenamiento.
        
        Args:
            state (np.array): Estado actual.
            action (int): Acción tomada.
            reward (float): Recompensa recibida.
            next_state (np.array): Estado resultante.
            done (bool): True si el episodio terminó.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, explore=True):
        """
        Selecciona una acción usando la estrategia epsilon-greedy.
        
        - **Exploración**: Con probabilidad epsilon, elige una acción al azar.
        - **Explotación**: Con probabilidad 1-epsilon, elige la mejor acción según la red.
        
        Args:
            state (np.array): Estado actual del entorno.
            explore (bool): Permite la exploración. Poner en False para la demostración.
        
        Returns:
            int: La acción seleccionada.
        """
        self.steps_done += 1
        
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size):
        """
        Entrena la red neuronal usando un lote de experiencias de la memoria.
        
        Este es el núcleo del algoritmo de aprendizaje DQN.
        
        Proceso:
        1. Muestrear un lote (batch) aleatorio de experiencias.
        2. Calcular los valores Q actuales (predicciones) con la red principal.
        3. Calcular los valores Q objetivo (targets) usando la red objetivo.
        4. Optimizar la red principal para minimizar la diferencia entre predicciones y targets.
        
        Args:
            batch_size (int): Número de experiencias a usar para el entrenamiento.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)
        
        # --- CÁLCULO DEL TARGET SEGÚN DQN ---
        # La red objetivo calcula el valor máximo del siguiente estado.
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        
        # Ecuación de Bellman para el target: R + gamma * max_Q(s', a')
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para prevenir gradientes explosivos y estabilizar.
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        self.optimizer.step()

        # Decaimiento de epsilon para reducir la exploración.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        """
        Carga los pesos de un modelo entrenado desde un archivo.
        
        Args:
            name (str): Ruta al archivo de pesos del modelo (.pth).
        """
        self.model.load_state_dict(torch.load(name))
        self.update_target_network()

    def save(self, name):
        """
        Guarda los pesos del modelo actual en un archivo.
        
        Args:
            name (str): Ruta donde se guardará el archivo de pesos (.pth).
        """
        torch.save(self.model.state_dict(), name)