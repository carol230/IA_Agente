# train_ga.py
"""
Implementación completa de algoritmo genético para entrenar agentes de IA.

Este módulo implementa un algoritmo genético completo que evoluciona poblaciones
de agentes para resolver el problema de recolección de frutas. El algoritmo
simula la evolución natural mediante selección, cruzamiento y mutación.

Proceso evolutivo:
1. Inicialización: Crear población aleatoria de agentes
2. Evaluación: Probar cada agente en escenarios aleatorios  
3. Selección: Elegir los mejores agentes como padres
4. Cruzamiento: Combinar genes de padres para crear hijos
5. Mutación: Introducir variación aleatoria en los genes
6. Reemplazo: Formar nueva generación con elitismo
7. Repetir hasta convergencia

Características del algoritmo:
- Evaluación en escenarios aleatorios para robustez
- Elitismo para preservar mejores soluciones
- Mutación gaussiana para exploración controlada
- Cruzamiento uniforme para recombinación equilibrada

"""

import torch
import numpy as np
from environment import GridEnvironment
from agent_ga import Agent, AgentNetwork
import random

# HIPERPARÁMETROS DEL ALGORITMO GENÉTICO
"""
Configuración de parámetros evolutivos que controlan el comportamiento
del algoritmo genético. Estos valores han sido ajustados empíricamente
para balancear exploración vs. explotación.
"""
POPULATION_SIZE = 100    # Tamaño de la población por generación
NUM_GENERATIONS = 500    # Número total de generaciones a evolucionar
MUTATION_RATE = 0.05     # Probabilidad de mutación por gen (5%)
ELITISM_COUNT = 25       # Mejores agentes que pasan directamente (25%)

GRID_SIZE = 5           # Tamaño del entorno de evaluación

def create_initial_population():
    """
    Crea la población inicial de agentes con genes aleatorios.
    
    Genera una población de agentes donde cada uno tiene pesos de red neuronal
    inicializados aleatoriamente según la inicialización por defecto de PyTorch.
    Esta diversidad inicial es crucial para el éxito del algoritmo genético.
    
    Returns:
        list: Lista de POPULATION_SIZE agentes con genes aleatorios
        
    Note:
        La diversidad genética inicial determina el espacio de búsqueda
        que el algoritmo puede explorar durante la evolución.
    """
    return [Agent() for _ in range(POPULATION_SIZE)]

def evaluate_fitness(population, env):
    """
    Evalúa el fitness de cada agente en la población.
    
    Cada agente se prueba en un escenario aleatorio generado dinámicamente.
    La variabilidad en los escenarios asegura que los agentes desarrollen
    estrategias robustas y generalizables en lugar de sobreajustarse a
    configuraciones específicas.
    
    Proceso de evaluación:
    1. Generar escenario aleatorio (posiciones, frutas, venenos)
    2. Ejecutar agente por máximo 50 pasos
    3. Acumular recompensas totales como fitness
    4. Repetir para todos los agentes de la población
    
    Args:
        population (list): Lista de agentes a evaluar
        env (GridEnvironment): Entorno de evaluación
        
    Note:
        El fitness se calcula como la suma total de recompensas obtenidas
        durante el episodio, incluyendo penalizaciones por movimientos,
        recompensas por frutas y penalizaciones por venenos.
    """
    for agent in population:
        # GENERACIÓN DE ESCENARIO ALEATORIO
        # Número variable de frutas (1-4) para diversidad de dificultad
        num_fruits = np.random.randint(1, 5)
        
        # Crear lista de todas las posiciones posibles
        all_pos = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(all_pos)  # Barajar para selección aleatoria
        
        # Asignar posiciones sin solapamiento
        agent_pos = all_pos.pop()   # Posición inicial del agente
        fruit_pos = [all_pos.pop() for _ in range(num_fruits)]  # Posiciones de frutas
        poison_pos = [all_pos.pop() for _ in range(np.random.randint(1, 4))]  # 1-3 venenos

        # EJECUCIÓN DEL EPISODIO
        state = env.reset(agent_pos=agent_pos, fruit_pos=fruit_pos, poison_pos=poison_pos)
        total_reward = 0
        
        # Máximo 50 pasos para evitar episodios infinitos
        for _ in range(50):
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            
            # Terminar si el agente completa el objetivo
            if done:
                break
                
        # Asignar fitness como recompensa total acumulada
        agent.fitness = total_reward

def selection(population):
    """
    Selecciona los mejores agentes de la población para reproducción.
    
    Implementa selección elitista donde solo los agentes con mayor fitness
    se seleccionan como padres para la siguiente generación. Este método
    asegura que las características exitosas se preserven y propaguen.
    
    Estrategia de selección:
    - Ordenar población por fitness (mayor a menor)
    - Seleccionar el 20% superior como padres
    - Estos padres participarán en cruzamiento y algunos en elitismo
    
    Args:
        population (list): Población de agentes evaluados
        
    Returns:
        list: Los mejores agentes seleccionados para reproducción
        
    Note:
        Un porcentaje del 20% permite suficiente diversidad genética
        mientras mantiene presión selectiva hacia mejores soluciones.
    """
    # Ordenar por fitness descendente (mejores primero)
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Seleccionar el 20% superior de la población
    return population[:int(POPULATION_SIZE * 0.2)]

def crossover(parent1, parent2):
    """
    Crea un nuevo agente combinando genes de dos padres.
    
    Implementa cruzamiento uniforme donde cada parámetro (gen) del hijo
    se hereda aleatoriamente de uno de los dos padres. Este método mantiene
    bloques funcionales de la red mientras permite recombinación genética.
    
    Proceso de cruzamiento:
    1. Crear nuevo agente hijo
    2. Para cada parámetro de la red neuronal:
       - Probabilidad 50%: heredar del padre 1
       - Probabilidad 50%: heredar del padre 2
    3. Cargar genes combinados en el hijo
    
    Args:
        parent1 (Agent): Primer padre seleccionado
        parent2 (Agent): Segundo padre seleccionado
        
    Returns:
        Agent: Nuevo agente hijo con genes combinados
        
    Note:
        El cruzamiento uniforme preserva mejor las estructuras funcionales
        de las redes neuronales comparado con cruzamiento de un punto.
    """
    # Crear nuevo agente hijo
    child = Agent()
    
    # Obtener diccionarios de parámetros (genes) de los padres
    p1_genes = parent1.network.state_dict()
    p2_genes = parent2.network.state_dict()
    child_genes = child.network.state_dict()

    # Cruzamiento uniforme: cada gen se hereda aleatoriamente
    for key in child_genes.keys():
        # 50% probabilidad de heredar cada gen de cada padre
        if random.random() < 0.5:
            child_genes[key] = p1_genes[key].clone()  # Heredar del padre 1
        else:
            child_genes[key] = p2_genes[key].clone()  # Heredar del padre 2
    
    # Cargar genes combinados en la red del hijo
    child.network.load_state_dict(child_genes)
    return child

def mutate(agent):
    """
    Introduce variación genética aleatoria en un agente.
    
    Implementa mutación gaussiana donde cada parámetro tiene una probabilidad
    MUTATION_RATE de ser alterado con ruido gaussiano. Esta mutación permite
    explorar nuevas regiones del espacio de búsqueda y evitar convergencia
    prematura a óptimos locales.
    
    Proceso de mutación:
    1. Para cada parámetro de la red neuronal:
       - Probabilidad MUTATION_RATE: agregar ruido gaussiano
       - Magnitud del ruido: distribución normal σ=0.1
    2. Recargar parámetros modificados en la red
    
    Args:
        agent (Agent): Agente a mutar
        
    Returns:
        Agent: El mismo agente con genes posiblemente mutados
        
    Note:
        La mutación gaussiana con σ=0.1 proporciona un balance entre
        exploración (nuevas soluciones) y explotación (preservar buenas soluciones).
    """
    child_genes = agent.network.state_dict()
    
    # Aplicar mutación a cada parámetro independientemente
    for key in child_genes.keys():
        if random.random() < MUTATION_RATE:
            # Agregar ruido gaussiano con desviación estándar 0.1
            noise = torch.randn_like(child_genes[key]) * 0.1
            child_genes[key] += noise
    
    # Recargar parámetros mutados en la red
    agent.network.load_state_dict(child_genes)
    return agent

if __name__ == "__main__":
    """
    Bucle principal del algoritmo genético.
    
    Ejecuta el proceso evolutivo completo a través de múltiples generaciones,
    implementando el ciclo: evaluación → selección → cruzamiento → mutación.
    Incluye elitismo para preservar las mejores soluciones y logging detallado
    del progreso evolutivo.
    """
    print("=" * 80)
    print("🧬 INICIANDO ENTRENAMIENTO CON ALGORITMO GENÉTICO 🧬")
    print("=" * 80)
    print(f"Parámetros de evolución:")
    print(f"🔹 Tamaño de población: {POPULATION_SIZE}")
    print(f"🔹 Generaciones: {NUM_GENERATIONS}")
    print(f"🔹 Tasa de mutación: {MUTATION_RATE*100}%")
    print(f"🔹 Elitismo: {ELITISM_COUNT} agentes")
    print("=" * 80)
    
    # INICIALIZACIÓN
    env = GridEnvironment()
    population = create_initial_population()
    
    print("🌱 Población inicial creada")
    print("🎯 Comenzando evolución...")
    print()

    # BUCLE EVOLUTIVO PRINCIPAL
    for gen in range(NUM_GENERATIONS):
        print(f"🔄 Generación {gen+1}/{NUM_GENERATIONS}")
        
        # FASE 1: EVALUACIÓN DE FITNESS
        evaluate_fitness(population, env)
        
        # FASE 2: SELECCIÓN DE PADRES
        parents = selection(population)
        
        # FASE 3: ANÁLISIS DE PROGRESO
        best_agent_of_gen = parents[0]  # El mejor agente de esta generación
        best_fitness = best_agent_of_gen.fitness
        avg_fitness = np.mean([agent.fitness for agent in population])
        
        # Logging del progreso evolutivo
        print(f"   📊 Mejor fitness: {best_fitness:.2f}")
        print(f"   📈 Fitness promedio: {avg_fitness:.2f}")
        print(f"   🏆 Mejora: {best_fitness - avg_fitness:.2f}")

        # FASE 4: PRESERVACIÓN DEL MEJOR AGENTE
        # Guardar genes del mejor agente de esta generación
        torch.save(best_agent_of_gen.network.state_dict(), "best_agent_genes.pth")
        print(f"   💾 Mejor agente guardado")

        # FASE 5: CREACIÓN DE NUEVA GENERACIÓN
        new_population = []
        
        # ELITISMO: Los mejores agentes pasan directamente
        new_population.extend(parents[:ELITISM_COUNT])
        print(f"   👑 {ELITISM_COUNT} elite preservados")

        # REPRODUCCIÓN: Llenar resto con descendencia
        offspring_count = 0
        while len(new_population) < POPULATION_SIZE:
            # Seleccionar dos padres aleatoriamente del pool de elite
            parent1, parent2 = random.sample(parents, 2)
            
            # Cruzamiento: combinar genes de padres
            child = crossover(parent1, parent2)
            
            # Mutación: introducir variación genética
            child = mutate(child)
            
            new_population.append(child)
            offspring_count += 1
            
        print(f"   👶 {offspring_count} descendientes creados")
        
        # Reemplazar población anterior
        population = new_population
        
        print("   ✅ Generación completada")
        print("-" * 60)

    # FINALIZACIÓN
    print("\n" + "=" * 80)
    print("🎉 ¡ENTRENAMIENTO EVOLUTIVO COMPLETADO! 🎉")
    print("=" * 80)
    print("📁 El mejor ADN evolutivo está guardado en 'best_agent_genes.pth'")
    print("🚀 Puedes usar este agente en los demostradores para ver su comportamiento")
    print("🧬 El agente ha evolucionado a través de", NUM_GENERATIONS, "generaciones")
    print("=" * 80)