from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as nng
from torch_geometric.utils import to_networkx, softmax
from torch_geometric.data import Data

import networkx as nx

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
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None,
        ) -> torch.tensor:

        '''
        forward pass through the node layer.

        returns updated node embeddings for the graph 
        '''

        # Perform Embedding
        if self.embed:
            nodes = F.leaky_relu(self.node_embedding_layer(x))
            neighbours, atom_batch_index, neighbour_counts = self.embed_neighbours(x, edge_index, edge_attr)
            
        else:
            nodes = x
            neighbours, atom_batch_index, neighbour_counts = self.get_neighbours(x, edge_index)

        # expand nodes to match neighbours.size()
        expanded = torch.concat([nodes[i].expand(index, nodes.shape[1]) for i, index in enumerate(neighbour_counts)])

        # get alignments and shizz
        joint_attributes = torch.concat([expanded, neighbours], axis=1)
        alignment = F.leaky_relu(self.alignment_layer(self.dropout(joint_attributes)))
        attentions = softmax(alignment, atom_batch_index)
        contexts = F.elu(
            nng.global_add_pool(
                torch.mul(attentions, self.context_layer(self.dropout(neighbours))), 
                atom_batch_index
                )
            )
        readout = F.relu(self.readout_layer(contexts, nodes))

        self.attentions = attentions
        self.node_embeddings = readout
        self.batch_index = atom_batch_index

        return readout

        

    def embed_neighbours(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None
        ) -> Tuple[torch.tensor, torch.tensor, List[int]]:

        '''
        embeds the neighbour descriptors (with edge_attr if not None)

        returns: 
            embedded neighbour tensor: tensor containing neighbour embeddings 
            atom_batch_index: indices of the atoms to which the neighbour belongs
            neighbour_counts: list of size #molecules where element i is the number of atoms in molecule i
        '''

        # graph of batch for easy access to node neighbours
        batch_graph = to_networkx(
            Data(edge_index=edge_index, num_nodes=len(x)),
            to_undirected=False
            )
        
        # list of neighbour indices and neighbour counts
        neighbour_indices, neighbour_counts = [
            (sorted(list(nx.all_neighbors(batch_graph, i))), len(list(nx.all_neighbors(batch_graph, i))))
            for i in range(x.shape[0])
            ]
        # list of neighbour attributes of size (node * n_node_neighbours for node in nodes) by node_attribute_dim 
        neighbour_attributes = torch.vstack([x[index] for index in neighbour_indices])
        # list of atom batch indices for softmax(alignment) so softmax is only completed across neighbours
        atom_batch_index = torch.concatenate([[i]*count for i, count in enumerate(neighbour_counts)])
        # list of atom_neighbours pairs for finding bond_attributes to concatenate to neighbour_attributes
        atom_neighbour_pairs = torch.concat([
            torch.tensor(list(batch_graph.in_edges(i)) + list(batch_graph.out_edges(i)))
            for i in range(x.shape[0])
        ], axis=0)
        # expanded edge_attributes for batch processes 
        expanded_edge_attr = edge_attr.expand(neighbour_attributes.shape[0], edge_attr.shape[0], edge_attr.shape[1])
        # bond masks to create bond_attribute tensor of size = neighbour_attributes.shape[0]
        # this is the most time consuming part of the process
        bond_masks = torch.vstack([torch.all(edge_index == pair.reshape(-1, 2).T, axis=0) for pair in atom_neighbour_pairs])
        # bond_attributes
        bond_attributes = expanded_edge_attr[bond_masks]

        return (
            F.leaky_relu(self.embed_neighbours(torch.concat([neighbour_attributes, bond_attributes], axis=1))),
            atom_batch_index,
            neighbour_counts
            )
    
    def get_neighbours(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        ) -> Tuple[torch.tensor, torch.tensor, List[int]]: 

        # graph of batch for easy access to node neighbours
        batch_graph = to_networkx(
            Data(edge_index=edge_index, num_nodes=len(x)),
            to_undirected=False
            )

        # list of neighbour indices and neighbour counts
        neighbour_indices, neighbour_counts = [
            (sorted(list(nx.all_neighbors(batch_graph, i))), len(list(nx.all_neighbors(batch_graph, i)))) 
            for i in range(x.shape[0])
            ]
        # list of neighbour attributes of size (node * n_node_neighbours for node in nodes) by node_attribute_dim 
        neighbour_attributes = torch.vstack([x[index] for index in neighbour_indices])
        # list of atom batch indices for softmax(alignment) so softmax is only completed across neighbours
        atom_batch_index = torch.concatenate([[i]*count for i, count in enumerate(neighbour_counts)])

        return (
            neighbour_attributes,
            atom_batch_index,
            neighbour_counts
        )


        

        









        
