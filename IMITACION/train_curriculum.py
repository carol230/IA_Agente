# train_curriculum.py
"""
Entrenamiento por curriculum learning para aprendizaje por imitación.

Este módulo implementa el entrenamiento de la red neuronal convolucional
utilizando curriculum learning con datasets de complejidad creciente.
El entrenamiento progresa desde escenarios simples (1 fruta) hasta
complejos (4 frutas), mejorando la convergencia y generalización.

Características:
    - Curriculum learning con 4 niveles de dificultad
    - Entrenamiento supervisado con pares estado-acción
    - Optimización Adam con learning rate adaptado
    - CrossEntropyLoss para clasificación de acciones
    - Progresión gradual de épocas por complejidad

Constantes:
    LEARNING_RATE: Tasa de aprendizaje para optimizador Adam (0.0005)
    BATCH_SIZE: Tamaño de lote para entrenamiento (128 muestras)
    CURRICULUM: Secuencia de datasets y épocas de entrenamiento

Flujo del entrenamiento:
    1. Lección 1: 1 fruta → 25 épocas (fundamentos básicos)
    2. Lección 2: 2 frutas → 30 épocas (navegación intermedia)
    3. Lección 3: 3 frutas → 40 épocas (planificación compleja)
    4. Lección 4: 4 frutas → 50 épocas (maestría y refinamiento)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from agent import AgentNetwork

# Hiperparámetros de entrenamiento
LEARNING_RATE = 0.0005  # Tasa de aprendizaje conservadora para estabilidad
BATCH_SIZE = 128        # Tamaño de lote balanceado para memoria y convergencia

# Curriculum de entrenamiento: (archivo_dataset, num_épocas)
CURRICULUM = [
    ("expert_data_1_fruit.pkl", 25),   # Nivel básico: conceptos fundamentales
    ("expert_data_2_fruits.pkl", 30),  # Nivel intermedio: decisiones múltiples
    ("expert_data_3_fruits.pkl", 40),  # Nivel avanzado: planificación compleja
    ("expert_data_4_fruits.pkl", 50)   # Nivel experto: refinamiento y maestría
]

if __name__ == "__main__":
    """
    Script principal de entrenamiento por curriculum learning.
    
    Implementa el entrenamiento secuencial de la red neuronal utilizando
    datasets de complejidad creciente. Cada lección del curriculum se
    enfoca en un nivel específico de dificultad, permitiendo al modelo
    aprender gradualmente conceptos más complejos.
    
    Proceso de entrenamiento:
        1. Inicialización del modelo, optimizador y función de pérdida
        2. Para cada lección del curriculum:
           a. Cargar dataset correspondiente
           b. Preparar DataLoader con batches mezclados
           c. Entrenar por número específico de épocas
           d. Monitorear pérdida promedio por época
        3. Guardar modelo final entrenado
    
    Beneficios del curriculum learning:
        - Convergencia más rápida y estable
        - Mejor generalización a nuevos escenarios
        - Reducción de overfitting a configuraciones específicas
        - Aprendizaje progresivo de conceptos complejos
    """
    # Inicializar componentes del entrenamiento
    model = AgentNetwork()                              # Red convolucional para predicción de acciones
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizador Adam para gradiente adaptativo
    criterion = nn.CrossEntropyLoss()                  # Función de pérdida para clasificación multiclase

    # Ejecutar curriculum learning secuencial
    for i, (dataset_file, num_epochs) in enumerate(CURRICULUM):
        print(f"\n--- Iniciando Lección {i+1}/{len(CURRICULUM)}: {dataset_file} ---")
        
        # Cargar dataset de demostración experta
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
        
        # Preparar datos para entrenamiento
        # Separar estados (entrada CNN) y acciones (etiquetas de clasificación)
        states = torch.FloatTensor(np.array([item[0] for item in data]))   # Estados visuales (3, 5, 5)
        actions = torch.LongTensor(np.array([item[1] for item in data]))   # Índices de acciones (0-3)
        
        # Crear DataLoader para entrenamiento por lotes
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(states, actions),
            batch_size=BATCH_SIZE, 
            shuffle=True  # Mezclar datos para evitar patrones de orden
        )

        # Entrenar modelo en el dataset actual
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Acumulador de pérdida para la época
            
            # Procesar todos los lotes del dataset
            for batch_states, batch_actions in dataloader:
                # Paso hacia adelante: predicción del modelo
                optimizer.zero_grad()           # Limpiar gradientes acumulados
                outputs = model(batch_states)   # Inferencia: estados → valores de acción
                
                # Calcular pérdida de clasificación
                loss = criterion(outputs, batch_actions)  # CrossEntropy entre predicción y etiqueta
                
                # Retropropagación y optimización
                loss.backward()    # Calcular gradientes por backpropagation
                optimizer.step()   # Actualizar pesos de la red
                
                # Acumular pérdida para monitoreo
                epoch_loss += loss.item()
            
            # Reporte de progreso por época
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Época {epoch+1}/{num_epochs}, Pérdida: {avg_loss:.4f}")

    # Guardar modelo entrenado final
    torch.save(model.state_dict(), "imitacion_model.pth")
    print("\n¡Entrenamiento por currículo completado! Modelo final guardado.")