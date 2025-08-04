# generate_data.py
import numpy as np
import random
import pickle
from environment import GridEnvironment
from a_star_solver import a_star_search

GRID_SIZE = 5

def get_action(from_pos, to_pos):
    delta = np.array(to_pos) - np.array(from_pos)
    if delta[0] == -1: return 0
    if delta[0] == 1: return 1
    if delta[1] == -1: return 2
    if delta[1] == 1: return 3
    return -1

def generate_expert_data_for_n_fruits(num_fruits, num_samples, output_file):
    env = GridEnvironment()
    expert_data = []
    print(f"Generando {num_samples} muestras para {num_fruits} fruta(s)...")
    
    generated_episodes = 0
    while len(expert_data) < num_samples:
        generated_episodes += 1
        num_poisons = np.random.randint(2, 5)
        all_pos = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(all_pos)
        
        agent_p = all_pos.pop()
        fruit_p = [all_pos.pop() for _ in range(num_fruits)]
        poison_p = [all_pos.pop() for _ in range(num_poisons)]
        
        env.reset(agent_pos=agent_p, fruit_pos=fruit_p, poison_pos=poison_p)

        for _ in range(50):
            if not env.fruit_pos: break
            
            agent_pos = env.agent_pos
            distances = [np.linalg.norm(agent_pos - f) for f in env.fruit_pos]
            goal_fruit = env.fruit_pos[np.argmin(distances)]
            path = a_star_search(GRID_SIZE, agent_pos, goal_fruit, env.poison_pos)

            if path and len(path) > 0:
                action = get_action(agent_pos, path[0])
                state = env.get_state()
                expert_data.append((state, action))
                env.step(action)
            else:
                break
        
        if generated_episodes % 200 == 0:
            print(f"  Partidas procesadas: {generated_episodes}, Muestras actuales: {len(expert_data)}")

    with open(output_file, "wb") as f:
        pickle.dump(expert_data[:num_samples], f)
    print(f"Dataset '{output_file}' creado con {len(expert_data[:num_samples])} muestras.")

if __name__ == "__main__":
    generate_expert_data_for_n_fruits(1, 4000, "expert_data_1_fruit.pkl")
    generate_expert_data_for_n_fruits(2, 4000, "expert_data_2_fruits.pkl")
    generate_expert_data_for_n_fruits(3, 4000, "expert_data_3_fruits.pkl")
    generate_expert_data_for_n_fruits(4, 5000, "expert_data_4_fruits.pkl")
    print("\nTodos los datasets del currículo han sido generados.")