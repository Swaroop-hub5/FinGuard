import torch
from data_gen import generate_synthetic_graph
from model import FincrimeGNN
import os

device = torch.device('cpu')

def train_model():
    # 1. Prepare Data
    data, G = generate_synthetic_graph()
    
    # Simple split
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * data.num_nodes)] = True
    test_mask = ~train_mask
    
    # 2. Initialize Model
    # Input features = 3 (Age, Amt, Risk), Classes = 2 (Safe, Fraud)
    model = FincrimeGNN(in_channels=3, hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()

    print("Starting training...")
    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            pred = out.argmax(dim=1)
            correct = (pred[test_mask] == data.y[test_mask]).sum()
            acc = int(correct) / int(test_mask.sum())
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    # 3. Save Artifacts
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'gnn_model.pth'))
    torch.save(data, os.path.join(MODEL_DIR, 'graph_data.pt'))
    
    
    '''if not os.path.exists('../models'):
        os.makedirs('../models')
    
    torch.save(model.state_dict(), '../models/gnn_model.pth')
    
    # Save graph data for the UI to load
    torch.save(data, '../models/graph_data.pt')
    print("Model and Data saved successfully.")'''

if __name__ == "__main__":
    train_model()