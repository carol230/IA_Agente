import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Red convolucional mejorada ---
class AgentNetwork(nn.Module):
    def __init__(self, h=5, w=5, outputs=4):
        super(AgentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        def conv2d_out(size, kernel=3, stride=1, padding=1):
            return (size + 2 * padding - kernel) // stride + 1

        convw = conv2d_out(conv2d_out(conv2d_out(w)))
        convh = conv2d_out(conv2d_out(conv2d_out(h)))
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- Agente con red y fitness ---
class Agent:
    def __init__(self):
        self.network = AgentNetwork()
        self.fitness = 0

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return torch.argmax(q_values).item()

    def load_genes(self, filepath):
        self.network.load_state_dict(torch.load(filepath))
