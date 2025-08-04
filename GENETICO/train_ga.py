# train_ga.py
import torch
import numpy as np
from environment import GridEnvironment
from agent_ga import Agent, AgentNetwork
import random

# --- Hiperparámetros Evolutivos ---
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.05  # 5% de probabilidad de que un gen mute
ELITISM_COUNT = 25    # Número de los mejores agentes que pasan directamente

GRID_SIZE = 5

def create_initial_population():
    return [Agent() for _ in range(POPULATION_SIZE)]

def evaluate_fitness(population, env):
    for agent in population:
        # Crea un escenario aleatorio para evaluar a cada agente
        num_fruits = np.random.randint(1, 5)
        all_pos = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(all_pos)
        agent_pos = all_pos.pop()
        fruit_pos = [all_pos.pop() for _ in range(num_fruits)]
        poison_pos = [all_pos.pop() for _ in range(np.random.randint(1, 4))]

        state = env.reset(agent_pos=agent_pos, fruit_pos=fruit_pos, poison_pos=poison_pos)
        total_reward = 0
        
        for _ in range(50): # 50 pasos por episodio
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        agent.fitness = total_reward

def selection(population):
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:int(POPULATION_SIZE * 0.2)] # Seleccionamos al 20% más apto

def crossover(parent1, parent2):
    child = Agent()
    p1_genes = parent1.network.state_dict()
    p2_genes = parent2.network.state_dict()
    child_genes = child.network.state_dict()

    for key in child_genes.keys():
        # Mezclamos los genes (pesos) de los padres
        if random.random() < 0.5:
            child_genes[key] = p1_genes[key]
        else:
            child_genes[key] = p2_genes[key]
    
    child.network.load_state_dict(child_genes)
    return child

def mutate(agent):
    child_genes = agent.network.state_dict()
    for key in child_genes.keys():
        if random.random() < MUTATION_RATE:
            # Añadimos una pequeña variación aleatoria al gen
            noise = torch.randn_like(child_genes[key]) * 0.1
            child_genes[key] += noise
    agent.network.load_state_dict(child_genes)
    return agent

if __name__ == "__main__":
    env = GridEnvironment()
    population = create_initial_population()

    for gen in range(NUM_GENERATIONS):
        evaluate_fitness(population, env)
        
        parents = selection(population)
        
        best_agent_of_gen = parents[0]
        best_fitness = best_agent_of_gen.fitness
        avg_fitness = np.mean([agent.fitness for agent in population])
        
        print(f"Generación: {gen+1}/{NUM_GENERATIONS}, Mejor Fitness: {best_fitness:.2f}, Fitness Promedio: {avg_fitness:.2f}")

        # Guardamos el ADN del mejor agente de la generación
        torch.save(best_agent_of_gen.network.state_dict(), "best_agent_genes.pth")

        new_population = []
        # Elitismo: los mejores pasan directamente a la siguiente generación
        new_population.extend(parents[:ELITISM_COUNT])

        # Llenamos el resto de la población con hijos
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population

    print("Entrenamiento evolutivo finalizado. El mejor ADN está en 'best_agent_genes.pth'")