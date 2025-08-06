# generate_data.py
"""
Generador de datos de demostración experta para aprendizaje por imitación.

Este módulo crea datasets de pares estado-acción obtenidos de un agente experto
que utiliza el algoritmo A* para navegación óptima. Los datos generados se
utilizan para entrenar agentes mediante aprendizaje supervisado, imitando
comportamiento experto en diferentes configuraciones de complejidad.

Funciones:
    get_action(from_pos, to_pos): Convierte movimiento posicional a índice de acción
    generate_expert_data_for_n_fruits(num_fruits, num_samples, output_file): 
        Genera dataset para configuración específica de frutas
        
Constantes:
    GRID_SIZE: Tamaño del entorno de cuadrícula (5x5)
"""
import numpy as np
import random
import pickle
from environment import GridEnvironment
from a_star_solver import a_star_search

# Configuración del entorno
GRID_SIZE = 5

def get_action(from_pos, to_pos):
    """
    Convierte un movimiento entre posiciones adyacentes en índice de acción.
    
    Calcula la diferencia vectorial entre posiciones y la mapea al índice
    de acción correspondiente. Utilizada para convertir el camino óptimo
    de A* en secuencia de acciones ejecutables por el agente.
    
    Args:
        from_pos (tuple/np.ndarray): Posición inicial (x, y)
        to_pos (tuple/np.ndarray): Posición objetivo (x, y)
    
    Returns:
        int: Índice de acción correspondiente al movimiento:
             0 = Arriba (decrementar x)
             1 = Abajo (incrementar x)  
             2 = Izquierda (decrementar y)
             3 = Derecha (incrementar y)
             -1 = Movimiento inválido (no adyacente)
    
    Example:
        >>> get_action((1, 1), (0, 1))  # Movimiento hacia arriba
        0
        >>> get_action((1, 1), (1, 2))  # Movimiento hacia derecha
        3
    
    Note:
        Solo funciona para posiciones adyacentes. Movimientos diagonales
        o de múltiples celdas retornan -1.
    """
    # Calcular vector de diferencia entre posiciones
    delta = np.array(to_pos) - np.array(from_pos)
    
    # Mapear diferencia a índice de acción
    if delta[0] == -1: return 0    # Arriba
    if delta[0] == 1: return 1     # Abajo
    if delta[1] == -1: return 2    # Izquierda
    if delta[1] == 1: return 3     # Derecha
    return -1  # Movimiento inválido

def generate_expert_data_for_n_fruits(num_fruits, num_samples, output_file):
    """
    Genera dataset de demostraciones expertas para configuración específica.
    
    Crea escenarios aleatorios con número fijo de frutas y utiliza A* para
    generar comportamiento experto óptimo. Implementa estrategia greedy de
    ir siempre a la fruta más cercana, creando datos de entrenamiento para
    aprendizaje por imitación con curriculum learning.
    
    Proceso de generación:
        1. Crear escenario aleatorio (agente, frutas, venenos)
        2. Calcular fruta más cercana al agente
        3. Usar A* para encontrar camino óptimo
        4. Ejecutar primer paso y registrar par (estado, acción)
        5. Repetir hasta completar episodio o fallar
        6. Continuar hasta obtener muestras suficientes
    
    Args:
        num_fruits (int): Número de frutas en cada escenario
        num_samples (int): Cantidad objetivo de muestras estado-acción
        output_file (str): Archivo pickle donde guardar el dataset
    
    Raises:
        IOError: Si no se puede escribir el archivo de salida
    
    Example:
        >>> generate_expert_data_for_n_fruits(2, 1000, "data_2_fruits.pkl")
        Generando 1000 muestras para 2 fruta(s)...
        Dataset 'data_2_fruits.pkl' creado con 1000 muestras.
    
    Note:
        - Venenos colocados aleatoriamente (2-4 por escenario)
        - Máximo 50 pasos por episodio para evitar bucles infinitos
        - Estrategia greedy: siempre ir a fruta más cercana (euclidiana)
        - Solo se registran acciones válidas (con camino A* factible)
    """
    # Inicializar entorno y contenedor de datos
    env = GridEnvironment()
    expert_data = []
    print(f"Generando {num_samples} muestras para {num_fruits} fruta(s)...")
    
    generated_episodes = 0
    while len(expert_data) < num_samples:
        generated_episodes += 1
        
        # Configurar escenario aleatorio
        num_poisons = np.random.randint(2, 5)  # 2-4 venenos por escenario
        
        # Generar posiciones únicas aleatorias
        all_pos = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(all_pos)
        
        # Asignar posiciones a elementos del entorno
        agent_p = all_pos.pop()
        fruit_p = [all_pos.pop() for _ in range(num_fruits)]
        poison_p = [all_pos.pop() for _ in range(num_poisons)]
        
        # Inicializar entorno con configuración generada
        env.reset(agent_pos=agent_p, fruit_pos=fruit_p, poison_pos=poison_p)

        # Simular episodio con comportamiento experto
        for _ in range(50):  # Máximo 50 pasos por episodio
            # Verificar condición de terminación por victoria
            if not env.fruit_pos: 
                break
            
            # Implementar estrategia greedy: ir a fruta más cercana
            agent_pos = env.agent_pos
            # Calcular distancias euclidianas a todas las frutas
            distances = [np.linalg.norm(agent_pos - f) for f in env.fruit_pos]
            # Seleccionar fruta más cercana como objetivo
            goal_fruit = env.fruit_pos[np.argmin(distances)]
            
            # Usar A* para encontrar camino óptimo al objetivo
            path = a_star_search(GRID_SIZE, agent_pos, goal_fruit, env.poison_pos)

            # Verificar si existe camino factible
            if path and len(path) > 0:
                # Convertir primer paso del camino a acción
                action = get_action(agent_pos, path[0])
                # Registrar par estado-acción para entrenamiento
                state = env.get_state()
                expert_data.append((state, action))
                # Ejecutar acción en el entorno
                env.step(action)
            else:
                # No hay camino factible: terminar episodio
                break
        
        # Reporte de progreso cada 200 episodios
        if generated_episodes % 200 == 0:
            print(f"  Partidas procesadas: {generated_episodes}, Muestras actuales: {len(expert_data)}")

    # Guardar dataset en archivo pickle
    with open(output_file, "wb") as f:
        pickle.dump(expert_data[:num_samples], f)
    print(f"Dataset '{output_file}' creado con {len(expert_data[:num_samples])} muestras.")

if __name__ == "__main__":
    """
    Script principal para generación de curriculum de datasets.
    
    Genera múltiples datasets con diferentes niveles de complejidad para
    implementar curriculum learning en el entrenamiento por imitación.
    Los datasets se ordenan por dificultad creciente (número de frutas).
    
    Curriculum generado:
        - 1 fruta: 4000 muestras (nivel básico)
        - 2 frutas: 4000 muestras (nivel intermedio bajo)
        - 3 frutas: 4000 muestras (nivel intermedio alto)
        - 4 frutas: 5000 muestras (nivel avanzado, más muestras)
    
    Beneficios del curriculum learning:
        - Aprendizaje gradual de complejidad creciente
        - Mejor convergencia del entrenamiento
        - Políticas más robustas y generalizables
        - Reducción de overfitting a configuraciones específicas
    """
    # Generar datasets con complejidad creciente
    generate_expert_data_for_n_fruits(1, 4000, "expert_data_1_fruit.pkl")
    generate_expert_data_for_n_fruits(2, 4000, "expert_data_2_fruits.pkl")
    generate_expert_data_for_n_fruits(3, 4000, "expert_data_3_fruits.pkl")
    generate_expert_data_for_n_fruits(4, 5000, "expert_data_4_fruits.pkl")
    print("\nTodos los datasets del currículo han sido generados.")