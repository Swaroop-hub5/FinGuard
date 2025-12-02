from fastapi import FastAPI, HTTPException
import torch
import sys
import os

device = torch.device('cpu')
# Add src to path to import model
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.model import FincrimeGNN

device = torch.device('cpu')

app = FastAPI(title="WiseGuard Fincrime API")

# Load Resources on Startup
MODEL_PATH = "models/gnn_model.pth"
DATA_PATH = "models/graph_data.pt"

try:
    data = torch.load(DATA_PATH)
    model = FincrimeGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Pre-compute predictions for fast lookup (in a real app, this would be on-demand)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        predictions = probs.argmax(dim=1)
        risk_scores = probs[:, 1] # Probability of being class 1 (Fraud)

except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "WiseGuard Graph API is running"}

@app.get("/predict/{node_id}")
def predict_node(node_id: int):
    if node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail="Node ID not found")
    
    return {
        "node_id": node_id,
        "is_fraud": bool(predictions[node_id].item()),
        "risk_score": float(risk_scores[node_id].item()),
        "features": data.x[node_id].tolist()
    }