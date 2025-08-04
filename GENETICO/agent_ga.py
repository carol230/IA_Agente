# agent_ga.py
import torch
import torch.nn as nn
import numpy as np

# La arquitectura de la red puede ser la misma CNN que ya teníamos.
# Es una buena forma de procesar la "visión" del agente.
class AgentNetwork(nn.Module):
    def __init__(self, h=5, w=5, outputs=4):
        super(AgentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# El Agente es ahora solo una cáscara con su red y una puntuación de fitness.
class Agent:
    def __init__(self):
        self.network = AgentNetwork()
        self.fitness = 0

    def choose_action(self, state):
        """Elige la acción sin exploración. Su comportamiento está 100% definido por sus genes."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.network(state_tensor)
        return torch.argmax(action_values).item()

    def load_genes(self, filepath):
        """Carga el ADN de un archivo para la demostración."""
        self.network.load_state_dict(torch.load(filepath))