# agent.py
"""
Implementación de agente con red neuronal convolucional para aprendizaje por imitación.

Este módulo define la arquitectura de red neuronal y la clase agente utilizadas
en el aprendizaje por imitación. La red procesa representaciones visuales del
entorno (grids 3D) y predice acciones óptimas imitando comportamiento experto.

Clases:
    AgentNetwork: Red neuronal convolucional para procesamiento de estados visuales
    Agent: Interfaz del agente que utiliza la red para toma de decisiones
"""
import torch
import torch.nn as nn

class AgentNetwork(nn.Module):
    """
    Red neuronal convolucional para procesamiento de estados de grid y predicción de acciones.
    
    Arquitectura diseñada específicamente para entornos de cuadrícula donde el estado
    se representa como imágenes de 3 canales (agente, frutas, venenos). Utiliza
    capas convolucionales para extracción de características espaciales seguidas
    de capas densas para predicción de acciones.
    
    Arquitectura:
        - Conv2D (3->16): Extracción de características básicas
        - Conv2D (16->32): Características de nivel medio
        - Flatten: Preparación para capas densas
        - FC (flattened->256): Representación de alto nivel
        - FC (256->4): Predicción de acciones (4 direcciones)
    
    Args:
        h (int): Altura del grid de entrada (default: 5)
        w (int): Ancho del grid de entrada (default: 5)
        outputs (int): Número de acciones posibles (default: 4)
    """
    def __init__(self, h=5, w=5, outputs=4):
        super(AgentNetwork, self).__init__()
        
        # Capas convolucionales para procesamiento espacial
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Función auxiliar para calcular tamaño de salida convolucional
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            """Calcula dimensión de salida después de operación convolucional."""
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # Calcular dimensiones después de capas convolucionales
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        # Capas densas para predicción final
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        """
        Propagación hacia adelante de la red neuronal.
        
        Procesa el estado visual del entorno a través de las capas convolucionales
        y densas para generar valores de acción. Utiliza ReLU como función de
        activación para introducir no-linealidad.
        
        Flujo de procesamiento:
            1. Conv1 + ReLU: Extracción de características básicas
            2. Conv2 + ReLU: Características de nivel medio
            3. Flatten: Conversión a vector 1D
            4. FC1 + ReLU: Representación de alto nivel
            5. FC2: Valores de acción finales (sin activación)
        
        Args:
            x (torch.Tensor): Estado del entorno con forma (batch_size, 3, h, w)
                             Canales: [agente, frutas, venenos]
        
        Returns:
            torch.Tensor: Valores de acción con forma (batch_size, 4)
                         Índices corresponden a [arriba, abajo, izquierda, derecha]
        """
        # Primera capa convolucional con activación ReLU
        x = nn.functional.relu(self.conv1(x))
        # Segunda capa convolucional con activación ReLU
        x = nn.functional.relu(self.conv2(x))
        # Aplanar para conexión con capas densas
        x = x.view(x.size(0), -1)
        # Primera capa densa con activación ReLU
        x = nn.functional.relu(self.fc1(x))
        # Capa de salida sin activación (valores de acción)
        return self.fc2(x)

class Agent:
    """
    Agente que utiliza red neuronal para toma de decisiones por imitación.
    
    Implementa la interfaz de agente que encapsula la red neuronal y proporciona
    métodos para selección de acciones y carga de modelos pre-entrenados.
    Diseñado para imitar comportamiento experto aprendido de datos de demostración.
    
    Attributes:
        network (AgentNetwork): Red neuronal convolucional para predicción de acciones
    """
    def __init__(self):
        """
        Inicializa el agente con red neuronal por defecto.
        
        Crea una instancia de AgentNetwork con parámetros estándar
        para entornos de grid 5x5 con 4 acciones posibles.
        """
        self.network = AgentNetwork()

    def choose_action(self, state):
        """
        Selecciona la acción óptima basada en el estado actual del entorno.
        
        Utiliza la red neuronal para evaluar el estado y selecciona la acción
        con mayor valor predicho. Implementa una política determinística
        (greedy) que siempre elige la mejor acción según el modelo.
        
        Proceso:
            1. Convierte estado a tensor PyTorch
            2. Agrega dimensión de batch (unsqueeze)
            3. Realiza inferencia sin gradientes
            4. Selecciona acción con mayor valor (argmax)
        
        Args:
            state (numpy.ndarray): Estado del entorno con forma (3, h, w)
                                  Canales: [agente, frutas, venenos]
        
        Returns:
            int: Índice de la acción seleccionada
                 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
        
        Note:
            Utiliza torch.no_grad() para optimizar inferencia y evitar
            construcción del grafo computacional durante evaluación.
        """
        # Convertir estado a tensor y agregar dimensión de batch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Realizar inferencia sin cálculo de gradientes
        with torch.no_grad():
            action_values = self.network(state_tensor)
        
        # Seleccionar acción con mayor valor predicho
        return torch.argmax(action_values).item()

    def load_model(self, filepath):
        """
        Carga pesos pre-entrenados en la red neuronal del agente.
        
        Permite cargar modelos entrenados mediante aprendizaje por imitación
        para utilizar políticas aprendidas de datos de demostración experta.
        Los pesos se cargan directamente en la red neuronal existente.
        
        Args:
            filepath (str): Ruta al archivo de modelo PyTorch (.pth)
                           que contiene los state_dict de la red
        
        Raises:
            FileNotFoundError: Si el archivo de modelo no existe
            RuntimeError: Si hay incompatibilidad en arquitectura de red
        
        Example:
            >>> agent = Agent()
            >>> agent.load_model('imitacion_model.pth')
            >>> action = agent.choose_action(current_state)
        
        Note:
            El modelo cargado debe tener la misma arquitectura que
            AgentNetwork para evitar errores de compatibilidad.
        """
        self.network.load_state_dict(torch.load(filepath))