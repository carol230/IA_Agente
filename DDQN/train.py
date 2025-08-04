# train.py
from environment import GridEnvironment
from agent import Agent
import numpy as np
import random

EPISODES = 25000 # Entrenar por 2000 "juegos"
GRID_SIZE = 5

if __name__ == "__main__":
    env = GridEnvironment(size=GRID_SIZE)
    state_shape = (3, GRID_SIZE, GRID_SIZE)
    action_size = 4
    agent = Agent(state_shape, action_size)
    
    # --- CAMBIO AQUÍ ---
    batch_size = 128 # Un batch size un poco más grande

    for e in range(EPISODES):
        # Generar un escenario aleatorio para cada episodio de entrenamiento
        # Esto es CLAVE para que el agente aprenda a generalizar
        num_fruits = np.random.randint(1, 5)
        num_poisons = np.random.randint(1, 4)
        
        # Asegurarse de que las posiciones no se repitan
        all_pos = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(all_pos)
        
        agent_pos = all_pos.pop()
        fruit_pos = [all_pos.pop() for _ in range(num_fruits)]
        poison_pos = [all_pos.pop() for _ in range(num_poisons)]

        state = env.reset(agent_pos=agent_pos, fruit_pos=fruit_pos, poison_pos=poison_pos)
        
        total_reward = 0
        for time in range(50):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Actualizar la red de destino cada N pasos
            if agent.steps_done % agent.update_target_every == 0:
                agent.update_target_network()

            if done:
                break
        
        print(f"Episodio: {e+1}/{EPISODES}, Puntuación: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # Ahora el aprendizaje se realiza después de cada paso, no al final del episodio
        agent.replay(batch_size)


        # Guardar el modelo cada 50 episodios
        if e % 50 == 0:
            agent.save("dqn_model.pth")

    print("Entrenamiento finalizado. Modelo guardado en 'dqn_model.pth'")