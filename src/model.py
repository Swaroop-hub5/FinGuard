import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FincrimeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FincrimeGNN, self).__init__()
        # SAGEConv aggregates information from neighbors
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Output Layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)