from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng

from torch_geometric.utils import softmax

class GraphLayer(nng.MessagePassing):

    '''
    torch_geometric module which updates a graph embeddings with node information + attention
    '''

    def __init__(
        self,
        dim: int,
        p_dropout: float=0.2
        ) -> None:

        super(GraphLayer, self).__init__()
        
        self.dim = dim
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(self.p_dropout)   

        self.alignment_layer = nn.Linear(2*self.dim, 1)
        self.context_layer = nn.Linear(self.dim, self.dim)
        self.readout_layer = nn.GRUCell(self.dim, self.dim)

    def forward(
        self,
        x: torch.tensor,
        batch_index: Optional[torch.tensor]=None,
        graph_nodes: Optional[torch.tensor]=None,
        ) -> torch.tensor:

        if batch_index is None:
            batch_index = torch.zeros(size=(x.shape[0]))

        if graph_nodes is None:
            graph_nodes = nng.global_add_pool(x, batch_index)
        else:
            graph_nodes = graph_nodes
        
        _, atom_counts = torch.unique(batch_index, return_counts=True)
        expanded = torch.concat([graph_nodes[i].expand(ac, graph_nodes.shape[1]) for i, ac in enumerate(atom_counts)], axis=0)

        joint_attributes = torch.concat([expanded, x], axis=1)
        alignment = F.leaky_relu(self.alignment_layer(self.dropout(joint_attributes)))
        attentions = softmax(alignment, batch_index)
        contexts = F.elu(
            nng.global_add_pool(
                torch.mul(attentions, self.context_layer(self.dropout(x))),
                batch_index
                )
            )
        readout = F.relu(self.readout_layer(contexts, graph_nodes))

        self.graph_nodes = graph_nodes
        self.joint_attributes = joint_attributes
        self.attentions = attentions
        self.graph_embeddings = readout
        self.batch_index = batch_index
        
        return readout
