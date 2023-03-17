from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng

from .node_layer import NodeLayer
from .graph_layer import GraphLayer

class AttentiveFP(nng.MessagePassing):

    def __init__(
        self,
        node_attribute_dim: int,
        edge_attribute_dim: Optional[int]=None,
        embedding_dim: int=256,
        n_node_layers: int=2,
        n_graph_layers: int=2,
        n_linear_layers: int=2,
        p_dropout: float=0.2
        ) -> None:

        super(AttentiveFP, self).__init__()

        self.embedding_dim = embedding_dim
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim

        self.node_layers = nn.ModuleList([
            NodeLayer(
                dim=embedding_dim, 
                embed=True, 
                node_attribute_dim=node_attribute_dim, 
                edge_attribute_dim=edge_attribute_dim,
                p_dropout=p_dropout
                )])

        self.node_layers.extend([
            NodeLayer(
                dim=embedding_dim, 
                p_dropout=p_dropout)
            for _ in range(n_node_layers-1)
         ])
        
        self.graph_layers = nn.ModuleList([
            GraphLayer(
                dim=embedding_dim,
                p_dropout=p_dropout)
            for _ in range(n_graph_layers)
         ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(n_linear_layers - 1)
         ])

        self.linear_layers.append(nn.Linear(embedding_dim, 1))

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(
        self,
        x: torch.tensor, 
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None,
        batch_index: Optional[torch.tensor]=None,
        ) -> torch.tensor:
        
        # Loop through node layers
        x = self.node_layers[0](x, edge_index, edge_attr)

        for mod in self.node_layers[1:]:
            x = mod(x, edge_index)

        # Loop through graph layers
        graph_embedding = self.graph_layers[0](x, batch_index)
        for mod in self.graph_layers[1:]:
            graph_embedding = mod(x, batch_index, graph_embedding)

        # Linear layers
        for mod in self.linear_layers[:-1]:
            graph_embedding = F.leaky_relu(mod(self.dropout(graph_embedding)))

        return self.linear_layers[-1](self.dropout(graph_embedding))
