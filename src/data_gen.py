import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from faker import Faker
import random

fake = Faker()

def generate_synthetic_graph(num_nodes=1000, num_edges=3000):
    print(f"Generating graph with {num_nodes} nodes...")
    G = nx.DiGraph()
    
    # 1. Generate Nodes (Accounts) with features
    # Features: [Account Age (days), Avg Transaction Amount, Risk Score (heuristic)]
    node_features = []
    for i in range(num_nodes):
        age = random.randint(1, 3650)
        avg_amt = random.uniform(10, 5000)
        risk_heuristic = random.uniform(0, 1)
        
        G.add_node(i, age=age, avg_amt=avg_amt, risk_init=risk_heuristic)
        node_features.append([age, avg_amt, risk_heuristic])

    # 2. Generate Random Legitimate Transactions
    for _ in range(num_edges):
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v, amount=random.uniform(10, 1000))

    # 3. Inject Fraud Rings (The "Signal" for our Model)
    # A ring is A -> B -> C -> A (Cyclic pattern often used in laundering)
    num_rings = 20
    fraud_indices = set()
    
    for _ in range(num_rings):
        ring_size = random.randint(3, 6)
        ring_nodes = random.sample(range(num_nodes), ring_size)
        
        for i in range(len(ring_nodes)):
            src = ring_nodes[i]
            dst = ring_nodes[(i + 1) % len(ring_nodes)] # Complete the cycle
            G.add_edge(src, dst, amount=random.uniform(9000, 9900)) # High amounts
            fraud_indices.add(src)
            fraud_indices.add(dst)

    # 4. Create Labels (1 = Fraud involved, 0 = Safe)
    labels = [1 if i in fraud_indices else 0 for i in range(num_nodes)]
    
    # 5. Convert to PyTorch Geometric Data format
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create PyG data object
    # We use NetworkX just to get edge_index easily
    pyg_data = from_networkx(G)
    pyg_data.x = x
    pyg_data.y = y
    
    # Normalize features
    pyg_data.x = (pyg_data.x - pyg_data.x.mean(dim=0)) / pyg_data.x.std(dim=0)

    print(f"Graph generated. Fraud nodes: {len(fraud_indices)}")
    return pyg_data, G