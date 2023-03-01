from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.nn as nng

class NodeLayer(nng.MessagePassing):

    '''
    torch_geometric module which updates node embeddings as detailed in X

    '''
    
    def __init__(
            self,
            node_attribute_dim: int,
            edge_attribute_dim: Optional[int]=None,
            embedding_dim: Optional[int]=None,
            p_dropout: float=0
        ) -> None:
        
        super(NodeLayer, self).__init__()

        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout

        if self.embedding_dim is not None:
            self.embed = True
            self.embed_atom_layer = nn.Linear(self.node_attribute_dim, self.embedding_dim)
            self.embed_neighbour_layer = \
                nn.Linear(self.node_attribute_dim + self.edge_attribute_dim, self.embedding_dim) \
                if self.edge_attribute_dim is not None \
                else nn.Linear(self.node_attribute_dim, self.embedding_dim)
        else:
            self.embed = False
            self.embedding_dim = self.node_attribute_dim
        
        self.alignment_layer = nn.Linear(2*self.embedding_dim, 1)
        self.context_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.readout = nn.GRUCell(self.embedding_dim, self.embedding_dim)

    def forward(
            self,
            x: Optional[torch.tensor]=None,
            edge_index: Optional[torch.tensor]=None,
            edge_attr: Optional[torch.tensor]=None,
            batch: Optional[torch.tensor]=None,
            on_batch: bool=False
        ) -> torch.tensor:

        







        
