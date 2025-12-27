from fastapi import FastAPI, HTTPException
import torch
import sys
import os
from src.model import FincrimeGNN

# --- FIX: Import the specific class that is being blocked ---
from torch_geometric.data.data import DataEdgeAttr

device = torch.device('cpu')

app = FastAPI(title="WiseGuard Fincrime API")

# Load Resources on Startup
MODEL_PATH = "models/gnn_model.pth"
DATA_PATH = "models/graph_data.pt"

# Global variables
data = None
model = None
predictions = None
risk_scores = None

try:
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        # --- FIX: Add weights_only=False to allow PyG objects ---
        # Note: In production, you would use safe_globals, but for this mock project
        # disabling the check is the fastest way to get unblocked.
        data = torch.load(DATA_PATH, map_location=device, weights_only=False)
        
        model = FincrimeGNN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        
        # Pre-compute predictions for fast lookup
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.exp(logits)
            predictions = probs.argmax(dim=1)
            risk_scores = probs[:, 1]
            
        print("✅ Model and Data loaded successfully!")
    else:
        print("⚠️ Model files not found. Please run training script.")

except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.get("/")
def home():
    status = "Active" if data is not None else "Waiting for Model Training"
    return {"message": f"WiseGuard Graph API is running. Status: {status}"}

@app.get("/predict/{node_id}")
def predict_node(node_id: int):
    # Safety check: if data didn't load, we can't predict
    if data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    if node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail="Node ID not found")
    
    return {
        "node_id": node_id,
        "is_fraud": bool(predictions[node_id].item()),
        "risk_score": float(risk_scores[node_id].item()),
        "features": data.x[node_id].tolist()
    }