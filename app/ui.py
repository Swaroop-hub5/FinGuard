import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
import os
import requests
import csv
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Analyst Workbench", layout="wide", page_icon="🛡️")

API_URL = os.getenv("API_URL", "http://localhost:8000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'models', 'graph_data.pt')

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; padding: 20px; border-radius: 10px; border: 1px solid #333;}
    .risk-reasoning {font-size: 16px; color: #cccccc; padding: 10px; border-left: 3px solid #ff4b4b; background-color: #262730;}
    .safe-reasoning {font-size: 16px; color: #cccccc; padding: 10px; border-left: 3px solid #00cc96; background-color: #262730;}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def log_feedback(node_id, model_pred, user_feedback, features):
    """Simulates writing to a Feature Store / Label Store for retraining."""
    # UPDATED PATH: Writes to the mounted volume
    FEEDBACK_FILE = "logs/feedback_log.csv"
    
    # Ensure the directory exists inside the container
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    with open(FEEDBACK_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "node_id", "model_prediction", "user_correction", "z_score_vol"])
        
        writer.writerow([
            datetime.now().isoformat(),
            node_id,
            "FRAUD" if model_pred else "SAFE",
            user_feedback,
            f"{features[1]:.2f}"
        ])

def generate_narrative(risk_score, features, risky_neighbors, total_neighbors):
    reasons = []
    if risk_score > 0.75: level = "CRITICAL"
    elif risk_score > 0.4: level = "SUSPICIOUS"
    else: level = "LOW RISK"

    if total_neighbors > 0:
        exposure = risky_neighbors / total_neighbors
        if exposure > 0.1: reasons.append(f"Directly connected to {risky_neighbors} known suspicious entities.")
        elif total_neighbors > 5: reasons.append("High degree of connectivity in a dense cluster.")
            
    amt_z = features[1]
    if amt_z > 1.5: reasons.append("Transaction volume is abnormally high (>1.5σ above avg).")
    elif amt_z < -1.0: reasons.append("Transaction volume is unusually low.")
        
    if not reasons: reasons.append("Account behavior fits standard legitimate profiles.")
    return level, reasons

@st.cache_resource
def load_graph_structure():
    if not os.path.exists(DATA_PATH): return None, None
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    G = to_networkx(data, to_undirected=False)
    return data, G

# --- INITIALIZATION ---
data, G = load_graph_structure()

if data is None:
    st.error("Graph data not found. Please train the model first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ FinGuard")
    st.markdown("### Case Management")
    node_id = st.number_input("Account ID", min_value=0, max_value=data.num_nodes-1, value=100)
    st.divider()
    st.caption(f"Network Size: {data.num_nodes} Accounts")
    
    # NEW: Download Button Section
    st.subheader("Data Export")
    FEEDBACK_FILE = "logs/feedback_log.csv"
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "rb") as f:
            st.download_button(
                label="📥 Download Feedback Log",
                data=f,
                file_name="analyst_feedback.csv",
                mime="text/csv"
            )
    else:
        st.caption("No feedback logged yet.")

# --- API INFERENCE ---
try:
    response = requests.get(f"{API_URL}/predict/{node_id}", timeout=2.0)
    if response.status_code == 200:
        result = response.json()
        is_fraud_val = result["is_fraud"]
        risk_score_val = result["risk_score"]
        features = result["features"]
    else:
        st.error(f"API Error: {response.status_code}")
        st.stop()
except Exception as e:
    st.error(f"❌ Connection Failed. Is the Docker API container running? \n\nError: {e}")
    st.stop()

neighbors = list(G.neighbors(node_id)) + list(G.predecessors(node_id))
risky_neighbors_count = sum([1 for nb in neighbors if data.y[nb] == 1])
level, reasons = generate_narrative(risk_score_val, features, risky_neighbors_count, len(neighbors))

# --- MAIN DASHBOARD ---
c1, c2, c3 = st.columns([3, 1, 1])
with c1: st.title(f"Investigation #{node_id}")
with c2: 
    if is_fraud_val: st.error(f"⚠️ {level}")
    else: st.success(f"✅ {level}")
with c3:
    st.write("**Verify Prediction:**")
    col_conf, col_deny = st.columns(2)
    if col_conf.button("👍 Correct"):
        log_feedback(node_id, is_fraud_val, "CORRECT", features)
        st.toast("Feedback logged!")
    if col_deny.button("👎 Incorrect"):
        log_feedback(node_id, is_fraud_val, "INCORRECT", features)
        st.toast("Feedback logged for retraining!")

st.divider()

reason_html = "".join([f"<li>{r}</li>" for r in reasons])
css_class = "risk-reasoning" if is_fraud_val else "safe-reasoning"
st.markdown(f"""<div class="{css_class}"><b>Model Reasoning:</b><ul>{reason_html}</ul></div>""", unsafe_allow_html=True)

st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Probability", f"{risk_score_val:.1%}", delta="High" if risk_score_val > 0.5 else "Safe", delta_color="inverse")
k2.metric("Network Exposure", f"{risky_neighbors_count}/{len(neighbors)}")
k3.metric("Volume (Z-Score)", f"{features[1]:.2f}")
k4.metric("Account Age (Norm)", f"{features[0]:.2f}")

st.divider()

col_graph, col_list = st.columns([2, 1])
with col_graph:
    st.subheader("Local Transaction Map")
    MAX_NEIGHBORS = 30
    display_nodes = neighbors[:MAX_NEIGHBORS] + [node_id]
    sub_G = G.subgraph(display_nodes)
    pos = nx.spring_layout(sub_G, seed=42)
    
    edge_x, edge_y = [], []
    for edge in sub_G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#555'), hoverinfo='none', mode='lines')

    node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node == node_id: color, size = '#FF4B4B', 30
        elif data.y[node] == 1: color, size = '#FFA500', 15
        else: color, size = '#00CC96', 15
            
        node_colors.append(color)
        node_sizes.append(size)
        node_text.append(f"ID: {node}")

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
    st.plotly_chart(fig, use_container_width=True)

with col_list:
    st.subheader("Neighbor List")
    if neighbors:
        neighbor_data = [{"ID": n, "Status": "Unknown"} for n in neighbors[:10]]
        st.dataframe(pd.DataFrame(neighbor_data), hide_index=True, use_container_width=True)