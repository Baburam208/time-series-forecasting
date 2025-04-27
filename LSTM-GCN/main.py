import torch
import numpy as np

from model import ASTGCNmodel

# Parameters
B = 1  # Batch size
N_nodes = 38  # Number of nodes
F_in = 5  # Number of input features
T_in = 12  # Temporal dimension

# Create the feature tensor x
x = torch.rand((B, N_nodes, F_in, T_in))

# Create edge_index (38 nodes in a fully connected graph)
# Use a simple fully connected graph for demonstration
edge_index = torch.tensor([
    [i, j] for i in range(N_nodes) for j in range(N_nodes) if i != j
], dtype=torch.long).T  # Shape: (2, num_edges)

# Create edge weights for the edges in edge_index
edge_weight = torch.rand(edge_index.shape[1])  # Random weights for each edge

nb_block: int = 4
in_channels: int = 5
K: int = 3
nb_chev_filter: int = 64
nb_time_filter: int = 64
time_strides: int = 2
num_for_predict: int = 1
len_input: int = 12
num_of_vertices: int = 38

model = ASTGCNmodel(nb_block,
                    in_channels,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    time_strides,
                    num_for_predict,
                    len_input,
                    num_of_vertices)

out = model(x, edge_index)

print(f"{type(out)}")
print(f"{out.shape = }")
