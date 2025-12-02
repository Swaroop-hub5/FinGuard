import torch

try:
    # Load the saved data
    data = torch.load("models/graph_data.pt")

    # Find indices where label is 1 (Fraud)
    fraud_nodes = (data.y == 1).nonzero(as_tuple=True)[0]

    print("--------------------------------------------------")
    print(f"Total Fraud Nodes found: {len(fraud_nodes)}")
    print("Here are 10 IDs you can test in the UI:")
    print(fraud_nodes[:10].tolist())
    print("--------------------------------------------------")

except FileNotFoundError:
    print("Error: Could not find 'models/graph_data.pt'. Make sure you trained the model first!")