# agent_ga.py
"""
Implementación de un agente basado en algoritmos genéticos para el entorno de recolección de frutas.

Este módulo define la arquitectura de red neuronal y la clase agente utilizados en el enfoque
de algoritmos genéticos. A diferencia del DQN que aprende mediante gradientes, este agente
evoluciona sus pesos mediante selección natural, mutación y cruzamiento.

Componentes principales:
- AgentNetwork: Red neuronal convolucional para procesar el estado visual
- Agent: Wrapper que contiene la red y maneja la evaluación de fitness

El agente procesa el estado del entorno (representado como una imagen de 3 canales)
y produce directamente acciones sin necesidad de aprendizaje por refuerzo.

"""

import torch
import torch.nn as nn
import numpy as np

# La arquitectura de la red puede ser la misma CNN que ya teníamos.
# Es una buena forma de procesar la "visión" del agente.
class AgentNetwork(nn.Module):
    """
    Red neuronal convolucional para el agente genético.
    
    Esta red procesa la representación visual del entorno (estado como imagen de 3 canales)
    y produce valores de acción para las 4 direcciones posibles. La arquitectura utiliza
    capas convolucionales para extraer características espaciales, seguidas de capas
    densas para la toma de decisiones.
    
    Arquitectura:
    - Conv2D (3→16 canales) + ReLU
    - Conv2D (16→32 canales) + ReLU  
    - Flatten
    - Dense (→256) + ReLU
    - Dense (→4 acciones)
    
    Args:
        h (int): Altura de la cuadrícula de entrada (default: 5)
        w (int): Ancho de la cuadrícula de entrada (default: 5)
        outputs (int): Número de acciones posibles (default: 4)
    """
    def __init__(self, h=5, w=5, outputs=4):
        super(AgentNetwork, self).__init__()
        
        # Capas convolucionales para procesamiento espacial del estado visual
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            """
            Calcula el tamaño de salida después de una operación de convolución.
            Formula: (entrada + 2*padding - kernel_size) // stride + 1
            """
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # Calcular dimensiones para la capa lineal después de las convoluciones
        # Como usamos padding=1 y kernel=3, las dimensiones se mantienen iguales
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32  # 32 es el número de canales de salida de conv2
        
        # Capas densas para la toma de decisiones
        # Capas densas para la toma de decisiones
        self.fc1 = nn.Linear(linear_input_size, 256)  # Capa oculta con 256 neuronas
        self.fc2 = nn.Linear(256, outputs)            # Capa de salida con 4 acciones

    def forward(self, x):
        """
        Propagación hacia adelante de la red neuronal.
        
        Procesa el estado visual del entorno a través de las capas convolucionales
        y densas para producir valores de acción.
        
        Args:
            x (torch.Tensor): Estado del entorno de forma (batch_size, 3, h, w)
                            - Canal 0: Posición del agente
                            - Canal 1: Posiciones de frutas  
                            - Canal 2: Posiciones de venenos
        
        Returns:
            torch.Tensor: Valores de acción de forma (batch_size, 4)
                         Cada valor representa la "utilidad" de una acción:
                         - Índice 0: Arriba
                         - Índice 1: Abajo
                         - Índice 2: Izquierda
                         - Índice 3: Derecha
        """
        # Primera capa convolucional + activación ReLU
        x = nn.functional.relu(self.conv1(x))
        
        # Segunda capa convolucional + activación ReLU
        x = nn.functional.relu(self.conv2(x))
        
        # Aplanar tensor para capas densas: (batch, channels*h*w)
        x = x.view(x.size(0), -1)
        
        # Primera capa densa + activación ReLU
        x = nn.functional.relu(self.fc1(x))
        
        # Capa de salida (sin activación, valores raw para argmax)
        return self.fc2(x)

# El Agente es ahora solo una cáscara con su red y una puntuación de fitness.
class Agent:
    """
    Wrapper del agente para algoritmos genéticos.
    
    Esta clase encapsula la red neuronal y proporciona la interfaz necesaria
    para el algoritmo genético. A diferencia de los agentes de RL, este agente
    no aprende durante la ejecución; su comportamiento está completamente
    determinado por los pesos de la red neuronal (sus "genes").
    
    El agente se evalúa mediante su fitness (rendimiento en el entorno),
    y los mejores agentes se seleccionan para reproducirse y crear la
    siguiente generación mediante:
    - Selección: Los mejores agentes tienen mayor probabilidad de reproducirse
    - Cruzamiento: Combinación de genes de dos padres
    - Mutación: Cambios aleatorios en los genes
    
    Attributes:
        network (AgentNetwork): Red neuronal que define el comportamiento del agente
        fitness (float): Puntuación de rendimiento en el entorno (mayor = mejor)
    """
    def __init__(self):
        """
        Inicializa un nuevo agente con red neuronal y fitness en cero.
        
        Los pesos de la red se inicializan aleatoriamente según la 
        inicialización por defecto de PyTorch. Estos pesos representan
        los "genes" del agente que evolucionarán con el tiempo.
        """
        self.network = AgentNetwork()
        self.fitness = 0

    def choose_action(self, state):
        """
        Selecciona una acción basada en el estado actual del entorno.
        
        El agente utiliza su red neuronal para evaluar el estado y selecciona
        la acción con el valor más alto (estrategia greedy). No hay exploración
        ya que el comportamiento del agente está completamente determinado por
        sus genes (pesos de la red).
        
        Este método es determinístico: dado el mismo estado y los mismos pesos,
        siempre producirá la misma acción. Esto es importante para la evaluación
        consistente del fitness durante la evolución.
        
        Args:
            state (np.array): Estado del entorno de forma (3, h, w)
                             - Canal 0: Posición del agente (1.0 donde está, 0.0 resto)
                             - Canal 1: Posiciones de frutas (1.0 donde hay frutas)
                             - Canal 2: Posiciones de venenos (1.0 donde hay venenos)
        
        Returns:
            int: Acción seleccionada:
                 - 0: Mover arriba (decrementar fila)
                 - 1: Mover abajo (incrementar fila)
                 - 2: Mover izquierda (decrementar columna)
                 - 3: Mover derecha (incrementar columna)
        """
        # Convertir estado NumPy a tensor PyTorch y agregar dimensión de batch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Evaluación sin gradientes (no hay backpropagation)
        with torch.no_grad():
            action_values = self.network(state_tensor)
        
        # Seleccionar acción con mayor valor Q (estrategia greedy)
        return torch.argmax(action_values).item()

    def load_genes(self, filepath):
        """
        Carga los "genes" (pesos de la red) desde un archivo.
        
        Utilizado para cargar agentes previamente evolucionados y demostrar
        su comportamiento. Los pesos representan el "ADN" del agente que
        determina completamente su comportamiento en el entorno.
        
        Este método es útil para:
        - Cargar el mejor agente de una evolución anterior
        - Demostrar el comportamiento de agentes elite
        - Continuar la evolución desde una generación guardada
        - Análisis y visualización del comportamiento aprendido
        
        Args:
            filepath (str): Ruta al archivo con los pesos del modelo
                           (normalmente un archivo .pth de PyTorch)
                           
        Raises:
            FileNotFoundError: Si el archivo no existe
            RuntimeError: Si los pesos no coinciden con la arquitectura
        """
        try:
            self.network.load_state_dict(torch.load(filepath))
            print(f"✅ Genes cargados exitosamente desde: {filepath}")
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {filepath}")
            raise
        except Exception as e:
            print(f"❌ Error cargando genes: {e}")
            raise