import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import FincrimeGNN

# --- CONFIG ---
st.set_page_config(page_title="WiseGuard | Analyst Workbench", layout="wide", page_icon="🛡️")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gnn_model.pth')
DATA_PATH = os.path.join(BASE_DIR, 'models', 'graph_data.pt')

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; padding: 20px; border-radius: 10px; border: 1px solid #333;}
    .risk-reasoning {font-size: 16px; color: #cccccc; padding: 10px; border-left: 3px solid #ff4b4b; background-color: #262730;}
    .safe-reasoning {font-size: 16px; color: #cccccc; padding: 10px; border-left: 3px solid #00cc96; background-color: #262730;}
</style>
""", unsafe_allow_html=True)

# --- CACHED LOADERS ---

@st.cache_resource
def load_system():
    if not os.path.exists(DATA_PATH): return None, None, None, None
    
    # ---------------- FIX START ----------------
    # Force map_location='cpu' to ensure cloud compatibility
    # weights_only=False is required for loading full Data objects
    data = torch.load(DATA_PATH, map_location=torch.device('cpu')) 
    # ---------------- FIX END ------------------
    
    G = to_networkx(data, to_undirected=False)
    
    device = torch.device('cpu')
    model = FincrimeGNN(in_channels=3, hidden_channels=64, out_channels=2)
    
    if os.path.exists(MODEL_PATH):
        # Apply the same fix for the model file
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        predictions = probs.argmax(dim=1)
        risk_scores = probs[:, 1]
        
    return data, G, predictions, risk_scores
'''
@st.cache_resource
def load_system():
    if not os.path.exists(DATA_PATH): return None, None, None, None
    
    data = torch.load(DATA_PATH)
    G = to_networkx(data, to_undirected=False)
    
    device = torch.device('cpu')
    model = FincrimeGNN(in_channels=3, hidden_channels=64, out_channels=2)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        predictions = probs.argmax(dim=1)
        risk_scores = probs[:, 1]
        
    return data, G, predictions, risk_scores'''

def generate_narrative(risk_score, features, risky_neighbors, total_neighbors):
    """
    Translates mathematical features into a plain English explanation for the analyst.
    """
    reasons = []
    
    # 1. Analyze Risk Score
    if risk_score > 0.75:
        level = "CRITICAL"
    elif risk_score > 0.4:
        level = "SUSPICIOUS"
    else:
        level = "LOW RISK"

    # 2. Analyze Network (The "GNN" part)
    if total_neighbors > 0:
        exposure = risky_neighbors / total_neighbors
        if exposure > 0.1:
            reasons.append(f"Directly connected to {risky_neighbors} known suspicious entities.")
        elif total_neighbors > 5:
            reasons.append("High degree of connectivity in a dense cluster.")
            
    # 3. Analyze Features (Z-Scores from data_gen.py)
    # features = [Age, Amt, Risk_Init]
    amt_z = features[1]
    if amt_z > 1.5:
        reasons.append("Transaction volume is abnormally high (>1.5σ above avg).")
    elif amt_z < -1.0:
        reasons.append("Transaction volume is unusually low.")
        
    if not reasons:
        reasons.append("Account behavior fits standard legitimate profiles.")
        
    return level, reasons

# --- INIT ---
data, G, predictions, risk_scores = load_system()

if data is None:
    st.error("Model artifacts not found.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ FinGuard")
    st.markdown("### Case Management")
    node_id = st.number_input("Account ID", min_value=0, max_value=data.num_nodes-1, value=100)
    
    st.divider()
    st.caption(f"Network Size: {data.num_nodes} Accounts")

# --- DATA PREP FOR UI ---
risk_score_val = float(risk_scores[node_id].item())
is_fraud_val = bool(predictions[node_id].item())
features = data.x[node_id].tolist()

# Get Neighbors
neighbors = list(G.neighbors(node_id)) + list(G.predecessors(node_id))
risky_neighbors_count = sum([1 for nb in neighbors if predictions[nb].item() == 1])
level, reasons = generate_narrative(risk_score_val, features, risky_neighbors_count, len(neighbors))

# --- MAIN DASHBOARD ---

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"Investigation #{node_id}")
with c2:
    if is_fraud_val:
        st.error(f"⚠️ {level}")
    else:
        st.success(f"✅ {level}")

# 1. INTELLIGENCE CARD (The "Why")
st.markdown("#### 🧠 Artificial Analyst Intelligence")
reason_html = "".join([f"<li>{r}</li>" for r in reasons])
css_class = "risk-reasoning" if is_fraud_val else "safe-reasoning"
st.markdown(f"""
<div class="{css_class}">
    <b>Model Reasoning:</b>
    <ul>{reason_html}</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# 2. KEY METRICS
k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Probability", f"{risk_score_val:.1%}", delta="High" if risk_score_val > 0.5 else "Safe", delta_color="inverse")
k2.metric("Network Exposure", f"{risky_neighbors_count}/{len(neighbors)}", help="Number of risky neighbors / Total neighbors")
k3.metric("Volume (Z-Score)", f"{features[1]:.2f}", help="Standard deviations from the average transaction amount")
k4.metric("Account Age (Norm)", f"{features[0]:.2f}")

st.divider()

# 3. VISUALIZATION
col_graph, col_list = st.columns([2, 1])

with col_graph:
    st.subheader("Transaction Map")
    
    # Subgraph logic
    subgraph_nodes = set(neighbors + [node_id])
    sub_G = G.subgraph(subgraph_nodes)
    pos = nx.spring_layout(sub_G, seed=42)
    
    # Edges
    edge_x, edge_y = [], []
    for edge in sub_G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#555'), hoverinfo='none', mode='lines')

    # Nodes
    node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Logic for coloring
        is_target = (node == node_id)
        is_risky = (predictions[node].item() == 1)
        
        if is_target:
            color = '#FF4B4B'  # Red
            size = 35
            label = f"<b>TARGET #{node}</b>"
        elif is_risky:
            color = '#FFA500'  # Orange
            size = 20
            label = f"Neighbor #{node} (RISK)"
        else:
            color = '#00CC96'  # Green
            size = 20
            label = f"Neighbor #{node} (SAFE)"
            
        node_colors.append(color)
        node_sizes.append(size)
        node_text.append(label)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(color=node_colors, size=node_sizes, line_width=2, line_color='white')
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    
    # FIX FOR WARNING: using 'width' instead of 'use_container_width'
    st.plotly_chart(fig, width="stretch")

with col_list:
    st.subheader("Risk Factors")
    if neighbors:
        neighbor_data = []
        for nb in neighbors:
            is_bad = predictions[nb].item() == 1
            neighbor_data.append({
                "ID": nb,
                "Status": "🚩 RISKY" if is_bad else "✅ SAFE",
                "Volume": f"{data.x[nb][1]:.1f}σ"
            })
        st.dataframe(pd.DataFrame(neighbor_data), hide_index=True, use_container_width=True)
    else:
        st.info("No network connections found.")