# train_curriculum.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from agent import AgentNetwork

LEARNING_RATE = 0.0005
BATCH_SIZE = 128

CURRICULUM = [
    ("expert_data_1_fruit.pkl", 25),
    ("expert_data_2_fruits.pkl", 30),
    ("expert_data_3_fruits.pkl", 40),
    ("expert_data_4_fruits.pkl", 50)
]

if __name__ == "__main__":
    model = AgentNetwork()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for i, (dataset_file, num_epochs) in enumerate(CURRICULUM):
        print(f"\n--- Iniciando Lección {i+1}/{len(CURRICULUM)}: {dataset_file} ---")
        
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
        
        states = torch.FloatTensor(np.array([item[0] for item in data]))
        actions = torch.LongTensor(np.array([item[1] for item in data]))
        
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(states, actions),
            batch_size=BATCH_SIZE, shuffle=True
        )

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/{num_epochs}, Pérdida: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "imitacion_model.pth")
    print("\n¡Entrenamiento por currículo completado! Modelo final guardado.")