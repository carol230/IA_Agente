# train.py
"""
Script de entrenamiento principal para el agente DDQN (Double Deep Q-Network).

Este módulo implementa el proceso completo de entrenamiento del agente DDQN para
el problema de recolección de frutas evitando venenos. El entrenamiento utiliza
generación aleatoria de escenarios para garantizar la generalización del agente.

Características principales del entrenamiento:
- Generación aleatoria de entornos para cada episodio
- Implementación completa del algoritmo DDQN
- Actualización periódica de la red objetivo
- Guardado automático del modelo durante el entrenamiento
- Monitoreo del progreso con métricas de rendimiento

El sistema está diseñado para entrenar un agente robusto capaz de manejar
una amplia variedad de configuraciones de entorno, desde escenarios simples
hasta configuraciones complejas con múltiples obstáculos y objetivos.

Algoritmo implementado:
- Double Deep Q-Network (DDQN) con replay buffer
- Exploración epsilon-greedy con decaimiento
- Actualización periódica de red objetivo
- Entrenamiento continuo con experiencias almacenadas
"""

from environment import GridEnvironment
from agent import Agent
import numpy as np
import random

# --- CONFIGURACIÓN DE ENTRENAMIENTO ---
"""Hiperparámetros principales del proceso de entrenamiento."""
EPISODES = 25000    # Número total de episodios de entrenamiento (juegos completos)
GRID_SIZE = 5       # Tamaño de la grilla del entorno (5x5 celdas)

if __name__ == "__main__":
    """
    Función principal que ejecuta el proceso completo de entrenamiento DDQN.
    
    Este bloque implementa el algoritmo de entrenamiento completo, incluyendo:
    - Inicialización del entorno y agente
    - Generación aleatoria de escenarios de entrenamiento
    - Bucle principal de entrenamiento con DDQN
    - Gestión de experiencias y actualización de redes
    - Monitoreo y guardado del progreso
    
    El entrenamiento utiliza curriculum learning implícito a través de la
    variabilidad aleatoria de escenarios, exponiendo al agente a una amplia
    gama de situaciones para mejorar la generalización.
    """
    
    # --- INICIALIZACIÓN DE COMPONENTES ---
    env = GridEnvironment(size=GRID_SIZE)              # Entorno de simulación
    state_shape = (3, GRID_SIZE, GRID_SIZE)           # Forma del estado: 3 canales x 5x5
    action_size = 4                                    # Número de acciones posibles (4 direcciones)
    agent = Agent(state_shape, action_size)           # Agente DDQN con arquitectura CNN
    
    # --- CONFIGURACIÓN DE HIPERPARÁMETROS ---
    batch_size = 128    # Tamaño del lote para entrenamiento de la red neural
                       # Un batch size mayor proporciona gradientes más estables

    # --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
    for e in range(EPISODES):
        # --- GENERACIÓN ALEATORIA DE ESCENARIOS ---
        """
        Cada episodio utiliza una configuración completamente aleatoria del entorno.
        Esta estrategia es FUNDAMENTAL para la generalización del agente, ya que
        evita el sobreajuste a configuraciones específicas y fuerza al agente
        a aprender estrategias robustas que funcionen en cualquier escenario.
        """
        
        # Determinar número aleatorio de elementos en el entorno
        num_fruits = np.random.randint(1, 5)    # Entre 1 y 4 frutas
        num_poisons = np.random.randint(1, 4)   # Entre 1 y 3 venenos
        
        # Generar posiciones únicas para todos los elementos
        # Esto previene superposiciones y garantiza configuraciones válidas
        all_pos = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(all_pos)  # Mezclar aleatoriamente todas las posiciones
        
        # Asignar posiciones únicas para cada elemento
        agent_pos = all_pos.pop()                           # Posición inicial del agente
        fruit_pos = [all_pos.pop() for _ in range(num_fruits)]   # Posiciones de frutas
        poison_pos = [all_pos.pop() for _ in range(num_poisons)] # Posiciones de venenos

        # Reiniciar entorno con la configuración generada
        state = env.reset(agent_pos=agent_pos, fruit_pos=fruit_pos, poison_pos=poison_pos)
        
        # --- EJECUCIÓN DEL EPISODIO ---
        """
        Cada episodio simula un juego completo donde el agente debe recolectar
        todas las frutas mientras evita los venenos. El límite de 50 pasos
        previene episodios infinitos y fuerza al agente a ser eficiente.
        """
        total_reward = 0
        
        # Bucle de pasos dentro del episodio (máximo 50 pasos)
        for time in range(50):
            # El agente elige una acción usando la política epsilon-greedy
            # Durante el entrenamiento, explora aleatoriamente con probabilidad epsilon
            action = agent.choose_action(state)
            
            # Ejecutar la acción en el entorno
            next_state, reward, done = env.step(action)
            
            # Almacenar la experiencia en el buffer de replay
            # Esta experiencia se usará más tarde para entrenar la red
            agent.remember(state, action, reward, next_state, done)
            
            # Actualizar estado y acumular recompensa
            state = next_state
            total_reward += reward

            # --- ACTUALIZACIÓN DE LA RED OBJETIVO ---
            """
            La red objetivo se actualiza periódicamente para estabilizar el entrenamiento.
            Esto es una característica clave del algoritmo DQN que previene la
            divergencia durante el entrenamiento.
            """
            if agent.steps_done % agent.update_target_every == 0:
                agent.update_target_network()

            # Terminar episodio si se completó el objetivo
            if done:
                break
        
        # --- MONITOREO DEL PROGRESO ---
        """
        Imprimir estadísticas del episodio para monitorear el progreso del entrenamiento.
        - Puntuación total: Indica qué tan bien está aprendiendo el agente
        - Epsilon: Muestra el balance actual entre exploración y explotación
        """
        print(f"Episodio: {e+1}/{EPISODES}, Puntuación: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # --- ENTRENAMIENTO DE LA RED NEURAL ---
        """
        El entrenamiento se realiza después de cada episodio usando experiencias
        almacenadas en el buffer de replay. Esto permite que el agente aprenda
        de experiencias pasadas, mejorando la eficiencia del aprendizaje.
        """
        agent.replay(batch_size)

        # --- GUARDADO PERIÓDICO DEL MODELO ---
        """
        Guardar el modelo cada 50 episodios para:
        - Prevenir pérdida de progreso en caso de interrupciones
        - Permitir evaluación de versiones intermedias
        - Facilitar la reanudación del entrenamiento si es necesario
        """
        if e % 50 == 0:
            agent.save("dqn_model.pth")

    # --- FINALIZACIÓN DEL ENTRENAMIENTO ---
    print("Entrenamiento finalizado. Modelo guardado en 'dqn_model.pth'")