# agent.py
"""
Implementación completa del agente DDQN (Double Deep Q-Network).

Este módulo contiene la implementación del algoritmo DDQN, una mejora del DQN clásico
que aborda el problema de sobreestimación de valores Q mediante el uso de dos redes
neuronales: una para selección de acciones y otra para evaluación de valores.

Características principales:
- Red neuronal convolucional optimizada para entornos de grilla
- Algoritmo DDQN con separación de selección y evaluación
- Memoria de replay extendida (50,000 experiencias)
- Técnicas de estabilización avanzadas
- Sistema robusto de guardado/carga de modelos

Algoritmo DDQN:
La innovación clave es el uso de dos redes para calcular targets:
1. Red principal: Selecciona la mejor acción del siguiente estado
2. Red objetivo: Evalúa el valor Q de esa acción seleccionada

Esto reduce significativamente la sobreestimación de valores Q que sufre DQN clásico,
resultando en un aprendizaje más estable y políticas más robustas.

Referencias:
- van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-learning"
- Mnih et al. (2015): "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. RED NEURONAL CONVOLUCIONAL PARA DDQN ---
class CNN_DQN(nn.Module):
    """
    Red neuronal convolucional especializada para DDQN en entornos de grilla.
    
    Esta arquitectura está optimizada para procesar estados representados como
    imágenes multi-canal, típicos en problemas de navegación espacial donde
    el estado se puede visualizar como una cuadrícula con diferentes tipos
    de elementos (agente, objetivos, obstáculos).
    
    Diseño arquitectónico:
    
    **Etapa Convolucional (Extracción de características):**
    - Conv1: 3→16 canales, kernel 3x3 → Detecta patrones básicos locales
    - Conv2: 16→32 canales, kernel 3x3 → Combina patrones en características complejas
    - ReLU en cada capa para introducir no-linealidad
    - Padding=1 preserva dimensiones espaciales
    
    **Etapa Completamente Conectada (Toma de decisiones):**
    - FC1: Procesa características extraídas (256 neuronas)
    - FC2: Genera valores Q para cada acción posible
    
    **Ventajas de esta arquitectura:**
    - Invarianza a traslaciones locales (convoluciones)
    - Reducción progresiva de parámetros vs redes totalmente conectadas
    - Capacidad de detectar patrones espaciales complejos
    - Escalabilidad a entornos de diferentes tamaños
    
    Args:
        h (int): Altura de la grilla de entrada
        w (int): Anchura de la grilla de entrada
        outputs (int): Número de acciones posibles (valores Q de salida)
    """
    
    def __init__(self, h, w, outputs):
        super(CNN_DQN, self).__init__()
        
        # --- CAPAS CONVOLUCIONALES PARA EXTRACCIÓN DE CARACTERÍSTICAS ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Entrada: 3 canales (agente, frutas, venenos)
        # Salida: 16 mapas de características
        # Kernel 3x3: Ventana de percepción local óptima para grillas pequeñas
        # Padding=1: Preserva dimensiones espaciales de entrada
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Entrada: 16 mapas de características de la capa anterior
        # Salida: 32 mapas de características más abstractas
        # Mayor profundidad permite detectar patrones más complejos
        
        # --- CÁLCULO DINÁMICO DE DIMENSIONES ---
        """
        Función auxiliar para calcular dimensiones después de convoluciones.
        Esencial para conectar correctamente las capas convolucionales
        con las capas completamente conectadas.
        
        Fórmula: output_size = (input_size + 2*padding - kernel_size) // stride + 1
        """
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # Aplicar la función de cálculo a ambas dimensiones espaciales
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32  # 32 canales de la última conv
        
        # --- CAPAS COMPLETAMENTE CONECTADAS PARA TOMA DE DECISIONES ---
        self.fc1 = nn.Linear(linear_input_size, 256)
        # Capa oculta densa que procesa las características extraídas
        # 256 neuronas: Balance entre capacidad expresiva y eficiencia computacional
        # Suficiente para capturar relaciones complejas entre características espaciales
        
        self.fc2 = nn.Linear(256, outputs)
        # Capa de salida que produce valores Q para cada acción
        # Sin función de activación (los valores Q pueden ser negativos)
        # Número de neuronas = número de acciones posibles

    def forward(self, x):
        """
        Propagación hacia adelante de la red neuronal.
        
        Implementa el flujo completo de información desde el estado de entrada
        hasta los valores Q de salida, aplicando las transformaciones necesarias
        para extraer características espaciales y generar estimaciones de valor.
        
        Args:
            x (torch.Tensor): Estado de entrada con forma (batch_size, 3, height, width)
                             - batch_size: Número de estados en el lote
                             - 3: Canales (agente, frutas, venenos)
                             - height, width: Dimensiones espaciales de la grilla
        
        Returns:
            torch.Tensor: Valores Q para cada acción con forma (batch_size, num_actions)
        
        Flujo de procesamiento:
        1. Convolución 1 + ReLU: Detección de patrones básicos
        2. Convolución 2 + ReLU: Extracción de características complejas
        3. Aplanamiento: Conversión de 2D a 1D para capas densas
        4. FC1 + ReLU: Procesamiento de alto nivel de características
        5. FC2: Generación de valores Q finales (sin activación)
        """
        # Primera capa convolucional con activación ReLU
        x = nn.functional.relu(self.conv1(x))
        
        # Segunda capa convolucional con activación ReLU
        x = nn.functional.relu(self.conv2(x))
        
        # Aplanar características espaciales para capas densas
        # Transforma tensor 4D (batch, canales, alto, ancho) → 2D (batch, características)
        x = x.view(x.size(0), -1)
        
        # Primera capa completamente conectada con activación ReLU
        x = nn.functional.relu(self.fc1(x))
        
        # Capa de salida sin activación (valores Q pueden ser negativos)
        return self.fc2(x)
    
# --- 2. AGENTE DDQN CON EXPERIENCE REPLAY Y TARGET NETWORK ---

class Agent:
    """
    Agente de Deep Q-Learning con arquitectura CNN para navegación en grilla.
    
    Implementa un agente de aprendizaje por refuerzo que utiliza una red neuronal
    convolucional para procesar estados espaciales y aprender una política óptima
    para navegar en un entorno de grilla, evitando venenos y recolectando frutas.
    
    Características principales:
    - CNN para procesamiento de estados espaciales (grilla 5x5)
    - Experience replay con buffer de memoria para estabilidad
    - Target network para cálculos de valores Q objetivo
    - Estrategia epsilon-greedy con decaimiento para exploración
    - Optimización Adam para entrenamiento eficiente
    
    Arquitectura del agente:
    1. Red principal: Entrenamiento y selección de acciones
    2. Red objetivo: Cálculos estables de valores Q futuro
    3. Buffer de experiencias: Almacena transiciones para replay
    4. Optimizador: Adam para actualización de pesos
    
    El agente mejora mediante:
    - Exploración inicial alta (epsilon=1.0) para descubrir el entorno
    - Decaimiento gradual hacia explotación (epsilon_min=0.01)
    - Entrenamiento con experiencias pasadas (experience replay)
    - Actualización periódica de la red objetivo para estabilidad
    """
    
    def __init__(self, state_shape, action_size):
        """
        Inicializa el agente DDQN con configuración optimizada para el entorno.
        
        Args:
            state_shape (tuple): Forma del estado (canales, altura, ancho)
                                Típicamente (3, 5, 5) para grilla con agente/frutas/venenos
            action_size (int): Número de acciones posibles (4: arriba, abajo, izq, der)
        
        Configuración de hiperparámetros:
        - memory: 50,000 experiencias para diversidad y estabilidad
        - gamma: 0.99 (alta importancia a recompensas futuras)
        - epsilon: 1.0→0.01 (exploración total a mínima)
        - epsilon_decay: 0.9995 (decaimiento gradual)
        - learning_rate: 0.0001 (ajuste fino y estable)
        - update_target_every: 5 (frecuencia de actualización de red objetivo)
        """
        self.state_shape = state_shape
        self.action_size = action_size
        
        # --- CONFIGURACIÓN DE EXPERIENCE REPLAY ---
        self.memory = deque(maxlen=50000)      
        # Buffer circular que almacena hasta 50,000 experiencias
        # Tamaño grande permite mayor diversidad de experiencias
        # Memoria circular: experiencias antiguas se eliminan automáticamente
        
        # --- PARÁMETROS DE APRENDIZAJE ---
        self.gamma = 0.99                     
        # Factor de descuento alto para valorar recompensas futuras
        # 0.99 significa que recompensas 100 pasos adelante valen ~37% del valor actual
        
        # --- ESTRATEGIA DE EXPLORACIÓN EPSILON-GREEDY ---
        self.epsilon = 1.0                    
        # Exploración inicial: 100% acciones aleatorias para descubrir entorno
        
        self.epsilon_min = 0.01               
        # Exploración mínima: siempre mantener 1% de acciones aleatorias
        # Evita quedar atrapado en mínimos locales
        
        self.epsilon_decay = 0.9995           
        # Decaimiento gradual: epsilon *= 0.9995 cada episodio
        # Transición suave de exploración a explotación
        
        # --- OPTIMIZACIÓN ---
        self.learning_rate = 0.0001           
        # Tasa de aprendizaje baja para entrenamiento estable y convergencia suave
        # Evita oscilaciones en la función de pérdida
        
        # --- ACTUALIZACIÓN DE RED OBJETIVO ---
        self.update_target_every = 5          
        # Frecuencia de actualización de la red objetivo (cada 5 entrenamientos)
        # Balance entre estabilidad y adaptación a nuevos pesos
        
        # --- INICIALIZACIÓN DE REDES NEURONALES ---
        h, w = state_shape[1], state_shape[2]  # Dimensiones de la grilla
        
        # Red principal: Se entrena continuamente con nuevas experiencias
        self.model = CNN_DQN(h, w, action_size)
        
        # Red objetivo: Proporciona valores Q estables para cálculos de objetivo
        self.target_model = CNN_DQN(h, w, action_size)
        
        # Inicializar red objetivo con mismos pesos que red principal
        self.update_target_network()
        
        # --- CONFIGURACIÓN DE ENTRENAMIENTO ---
        # Optimizador Adam: Adaptativo, eficiente para redes neuronales
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Función de pérdida: Error cuadrático medio para regresión de valores Q
        self.criterion = nn.MSELoss()
        
        # Contador de pasos para tracking de actualizaciones
        self.steps_done = 0

    def update_target_network(self):
        """
        Actualiza la red objetivo copiando pesos de la red principal.
        
        La red objetivo es fundamental para la estabilidad del entrenamiento:
        - Proporciona valores Q estables para calcular objetivos
        - Se actualiza menos frecuentemente que la red principal
        - Evita que los objetivos cambien constantemente durante entrenamiento
        
        Proceso:
        1. Copia completa de todos los parámetros de la red principal
        2. La red objetivo permanece fija hasta la próxima actualización
        3. Garantiza consistencia en los cálculos de valores Q objetivo
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en el buffer de memory replay.
        
        Experience replay es una técnica fundamental en Deep Q-Learning que:
        - Rompe correlaciones temporales entre experiencias consecutivas
        - Permite reutilizar experiencias valiosas múltiples veces
        - Mejora la eficiencia de uso de datos
        - Estabiliza el entrenamiento de la red neuronal
        
        Args:
            state (np.array): Estado actual del agente en la grilla
                             Forma (3, 5, 5) con canales para agente/frutas/venenos
            action (int): Acción tomada (0=arriba, 1=abajo, 2=izq, 3=der)
            reward (float): Recompensa recibida por la acción
                           +10 por fruta, -10 por veneno, -1 por movimiento
            next_state (np.array): Estado resultante después de la acción
            done (bool): True si el episodio terminó (todas frutas recogidas)
        
        El buffer circular (deque) gestiona automáticamente:
        - Eliminación de experiencias antiguas cuando se alcanza el límite
        - Mantener diversidad de experiencias para entrenamiento robusto
        - Acceso eficiente para muestreo aleatorio durante replay
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, explore=True):
        """
        Selecciona una acción usando estrategia epsilon-greedy.
        
        Implementa el balance crítico entre exploración y explotación:
        - Exploración: Necesaria para descubrir nuevas estrategias
        - Explotación: Usar conocimiento actual para maximizar recompensas
        
        Args:
            state (np.array): Estado actual del entorno (3, 5, 5)
            explore (bool): Si False, siempre usa la mejor acción conocida
                           Útil para evaluación sin exploración aleatoria
        
        Returns:
            int: Índice de acción seleccionada (0-3)
        
        Estrategia epsilon-greedy:
        - Probabilidad epsilon: Acción aleatoria (exploración)
        - Probabilidad (1-epsilon): Mejor acción según red neuronal (explotación)
        
        Progresión de epsilon:
        - Inicio: ε=1.0 → 100% exploración para mapear el entorno
        - Medio: ε~0.5 → Balance exploración/explotación
        - Final: ε=0.01 → 99% explotación, 1% exploración residual
        """
        self.steps_done += 1  # Contador para tracking de progreso
        
        # Exploración: acción aleatoria si epsilon lo determina y explore=True
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Explotación: usar red neuronal para encontrar mejor acción
        # Convertir estado a tensor PyTorch y agregar dimensión de batch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Inferencia sin calcular gradientes (más eficiente)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        
        # Seleccionar acción con mayor valor Q predicho
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size):
        """
        Entrena la red neuronal usando experiencias pasadas con Double DQN.
        
        Double DQN mejora el algoritmo DQN clásico al separar la selección
        y evaluación de acciones, reduciendo la sobrestimación sistemática
        de valores Q que puede llevar a políticas subóptimas.
        
        Diferencias DQN vs Double DQN:
        
        DQN Clásico:
        target = reward + gamma * max(target_network(next_state))
        Problema: La misma red selecciona y evalúa → sobrestimación
        
        Double DQN:
        best_action = argmax(main_network(next_state))     # Selección
        target = reward + gamma * target_network(next_state)[best_action]  # Evaluación
        Ventaja: Separación reduce sesgo de sobrestimación
        
        Args:
            batch_size (int): Tamaño del lote de experiencias para entrenamiento
                             Típicamente 32-64 para balance eficiencia/estabilidad
        
        Proceso de entrenamiento:
        1. Verificar que hay suficientes experiencias en memoria
        2. Muestrear batch aleatorio de experiencias
        3. Calcular valores Q actuales para estados del batch
        4. Aplicar lógica Double DQN para calcular objetivos
        5. Computar loss (MSE entre predicciones y objetivos)
        6. Backpropagation y actualización de pesos
        7. Decrecer epsilon (menos exploración)
        8. Aplicar clipping de gradientes para estabilidad
        """
        # No entrenar si memoria insuficiente
        if len(self.memory) < batch_size:
            return

        # --- MUESTREO ALEATORIO DE EXPERIENCIAS ---
        # Rompe correlaciones temporales y mejora generalización
        minibatch = random.sample(self.memory, batch_size)
        
        # Separar componentes de las experiencias en tensores
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1)

        # --- VALORES Q ACTUALES ---
        # Calcular Q-values para estados actuales usando red principal
        current_q_values = self.model(states).gather(1, actions)
        
        # --- LÓGICA DOUBLE DQN ---
        with torch.no_grad():  # No calcular gradientes para eficiencia
            # 1. Red principal SELECCIONA mejor acción para siguiente estado
            #    Usa conocimiento más actualizado para selección
            best_next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            
            # 2. Red objetivo EVALÚA el valor de la acción seleccionada
            #    Usa pesos más estables para evaluación consistente
            next_q_values_target = self.target_model(next_states).gather(1, best_next_actions)
        
        # --- CÁLCULO DE OBJETIVOS Q ---
        # Si episodio terminó (done=True), no hay valor futuro
        # target = reward + descuento * valor_futuro * (no_terminado)
        target_q_values = rewards + (self.gamma * next_q_values_target * (~dones))
        
        # --- ENTRENAMIENTO DE LA RED ---
        # Error cuadrático medio entre predicciones y objetivos
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimización con backpropagation
        self.optimizer.zero_grad()  # Limpiar gradientes previos
        loss.backward()             # Calcular gradientes
        
        # Clipping de gradientes para prevenir explosión
        # Limita gradientes a [-1, 1] para estabilidad numérica
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        
        self.optimizer.step()       # Actualizar pesos
        
        # --- DECAIMIENTO DE EXPLORACIÓN ---
        # Reducir epsilon gradualmente para transición exploración→explotación
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Carga un modelo pre-entrenado desde archivo.
        
        Args:
            name (str): Ruta al archivo .pth con los pesos del modelo
        
        Funcionalidad:
        - Carga pesos de la red principal desde archivo
        - Actualiza red objetivo para mantener consistencia
        - Permite continuar entrenamiento o hacer inferencia
        - Preserva arquitectura de red definida en __init__
        """
        self.model.load_state_dict(torch.load(name))
        self.update_target_network()  # Sincronizar red objetivo

    def save(self, name):
        """
        Guarda el modelo entrenado en un archivo.
        
        Args:
            name (str): Ruta donde guardar el archivo .pth
        
        Funcionalidad:
        - Guarda solo los pesos de la red principal (más compacto)
        - La red objetivo se puede reconstruir al cargar
        - Formato PyTorch estándar para compatibilidad
        - Permite reutilizar modelos entrenados
        """
        torch.save(self.model.state_dict(), name)