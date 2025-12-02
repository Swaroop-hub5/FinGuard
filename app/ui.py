import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
st.set_page_config(page_title="WiseGuard | Analyst Workbench", layout="wide", page_icon="🛡️")
API_URL = "http://127.0.0.1:8000"
DATA_PATH = "models/graph_data.pt"

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .high-risk {color: #ff4b4b; font-weight: bold;}
    .low-risk {color: #00cc96; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_graph_data():
    if os.path.exists(DATA_PATH):
        data = torch.load(DATA_PATH)
        # Convert to NetworkX for visualization
        G = to_networkx(data, to_undirected=False)
        return data, G
    return None, None

data, G = load_graph_data()

if not data:
    st.error("Model data not found. Please run 'src/train.py' first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ FinGuard")
    st.markdown("### Case Management")
    node_id = st.number_input("Target Account ID", min_value=0, max_value=data.num_nodes-1, value=0)
    
    if st.button("Run Investigation"):
        st.session_state['analyzed'] = True
    
    st.divider()
    st.info("💡 **Tip:** Use the helper script to find Fraud IDs (e.g., try ID from your terminal).")

# --- MAIN DASHBOARD ---
st.title(f"Investigation: Account #{node_id}")

if 'analyzed' in st.session_state:
    # 1. FETCH PREDICTION FROM API
    try:
        res = requests.get(f"{API_URL}/predict/{node_id}")
        if res.status_code == 200:
            result = res.json()
            risk_score = result['risk_score']
            is_fraud = result['is_fraud']
            features = result['features'] # [Age, AvgAmt, RiskInit] (Normalized)
        else:
            st.error("API Error")
            st.stop()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

    # 2. TOP METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Risk Score", f"{risk_score:.1%}", delta_color="inverse")
    
    with c2:
        status = "🚨 SUSPICIOUS" if is_fraud else "✅ CLEAR"
        st.metric("Model Classification", status)

    with c3:
        # Interpret Normalized Feature: Avg Transaction Amount
        # > 0 means above average, < 0 means below average
        amt_z = features[1]
        amt_label = "High Volume" if amt_z > 1 else ("Low Volume" if amt_z < -0.5 else "Average")
        st.metric("Volume Profile", amt_label, f"{amt_z:.2f} σ")

    with c4:
        # Degree (Number of connections)
        degree = G.degree[node_id]
        st.metric("Network Connections", f"{degree} Links")

    st.divider()

    # 3. ADVANCED GRAPH VISUALIZATION & NEIGHBOR ANALYSIS
    col_graph, col_data = st.columns([2, 1])

    with col_graph:
        st.subheader("Transaction Topology")
        
        # Get k=1 Hop Neighbors
        neighbors = list(G.neighbors(node_id)) + list(G.predecessors(node_id))
        subgraph_nodes = set(neighbors + [node_id])
        sub_G = G.subgraph(subgraph_nodes)
        
        # Layout
        pos = nx.spring_layout(sub_G, seed=42)
        
        # Edges
        edge_x, edge_y = [], []
        edge_colors = []
        
        for edge in sub_G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # If edge connects to our target, color it based on risk
            edge_colors.append('#555') 

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#666'),
            hoverinfo='none',
            mode='lines')

        # Nodes
        node_x, node_y = [], []
        node_colors = []
        node_text = []
        node_sizes = []
        
        for node in sub_G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Color Logic
            if node == node_id:
                node_colors.append('#FF4B4B') # Red for Target
                node_sizes.append(30)
                node_text.append(f"TARGET #{node}")
            else:
                # Retrieve risk from data.y for context (simulating looking up neighbor status)
                is_neighbor_fraud = data.y[node].item() == 1
                color = '#FFA500' if is_neighbor_fraud else '#00CC96' # Orange if fraud, Green if safe
                node_colors.append(color)
                node_sizes.append(20)
                node_text.append(f"Neighbor #{node}<br>Status: {'RISK' if is_neighbor_fraud else 'SAFE'}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line_width=2,
                line_color='white'))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=0),
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Target | 🟠 Risky Neighbor | 🟢 Safe Neighbor")

    with col_data:
        st.subheader("Neighbor Exposure")
        
        neighbor_data = []
        risky_neighbors = 0
        
        for nb in neighbors:
            # We peek at the ground truth for neighbors to simulate 'known' risk status
            is_bad = data.y[nb].item() == 1
            if is_bad: risky_neighbors += 1
            
            neighbor_data.append({
                "Account ID": nb,
                "Relationship": "Link",
                "Status": "RISK" if is_bad else "SAFE"
            })
            
        df = pd.DataFrame(neighbor_data)
        
        # Exposure Score
        exposure_rate = risky_neighbors / len(neighbors) if neighbors else 0
        st.metric("Exposure Rate", f"{exposure_rate:.0%}", 
                 help="Percentage of neighbors flagged as suspicious")
        
        if not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No direct neighbors found.")