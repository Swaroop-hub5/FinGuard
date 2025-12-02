# FinGuard: Graph Neural Network for Anti-Money Laundering (AML)

![Status](https://img.shields.io/badge/Status-Prototype-blue) ![Python](https://img.shields.io/badge/Python-3.9+-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red)

📸 Screenshots
![Preview](Screenshot.png)

**FinGuard** is an end-to-end Machine Learning platform designed to detect complex financial crime patterns, specifically **money laundering rings (smurfing)**, which traditional tabular models often miss.

It leverages **Graph Neural Networks (GraphSAGE)** to analyze the topology of transaction networks, classifying accounts as "Safe" or "Suspicious" based on both their features and their connections.

## 🚀 Key Features

* **Graph Neural Network (GNN):** Implements an inductive **GraphSAGE** model using `PyTorch Geometric` to detect cyclic transaction patterns.
* **Synthetic Data Engine:** Generates realistic transaction graphs with injected fraud rings (cliques) and variable node features (Account Age, Risk Score).
* **Real-time Inference API:** A **FastAPI** backend that serves model predictions and risk scores.
* **Analyst Workbench:** A **Streamlit** dashboard for Fincrime analysts to visualize subgraphs, inspect neighbor risk exposure, and interpret Z-score feature metrics.
* **Scalable Structure:** Modular code design separating data generation, training, and deployment logic.

## 🛠️ Tech Stack

* **ML Core:** PyTorch, PyTorch Geometric, NetworkX
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit, Plotly (for interactive graph viz)
* **Data Ops:** Pandas, NumPy, Faker

1. Clone the repository
git clone [https://github.com/yourusername/FinGuard.git](https://github.com/yourusername/FinGuard.git)
cd FinGuard
2. Create a Virtual Environment

Windows
python -m venv venv

.\venv\Scripts\activate

Mac/Linux
python3 -m venv venv

source venv/bin/activate

3. Install Dependencies
Note: This project is optimized for CPU usage to be lightweight.

Install PyTorch CPU first to avoid large downloads
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

Install remaining dependencies
pip install -r requirements.txt

## 🏃‍♂️ Usage Guide

Step 1: Train the Model

Generate synthetic data and train the GraphSAGE model.

python src/train.py

Output: Saves gnn_model.pth and graph_data.pt to the models/ directory.

Step 2: Start the Inference API

Launch the backend server to handle prediction requests.

uvicorn app.api:app --reload

API will run at http://127.0.0.1:8000. 

Step 3: Launch the Dashboard

Open a new terminal and start the Analyst Workbench.

streamlit run app/ui.py

The UI will open in your browser at http://localhost:8501.

## 🔍 How to Test

Since the data is synthetic, you need to find a specific node ID that is part of a "Fraud Ring" to see the detection capabilities in action.

Run the helper script to find a target:

Python

## Run in python shell
import torch

data = torch.load("models/graph_data.pt")

print((data.y == 1).nonzero(as_tuple=True)[0].tolist()[:5])

Copy one of these IDs and paste it into the Target Account ID field in the Dashboard sidebar.


## 🔮 Future Improvements
Dockerization: Containerize API and UI for easier deployment.

Explainability: Implement GNNExplainer to highlight exactly which edges contributed to the fraud score.

Database Integration: Replace file-based loading with Neo4j or AWS Neptune.