from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as nng
from torch_geometric.utils import softmax

class NodeLayer(nng.MessagePassing):

    '''
    torch_geometric module which updates node embeddings as detailed in X

    Args:

        node_attribute_dim (int): the expected dimension of the node attributes
        edge_attribute_dim (Optional[int]=None): the expected dimension of the edge_attributes. If None, no edge_attributes expected
        embedding_dim (Optional[int]=None): the size of the embedding dimension. If int is passed, the node and node neighbours are
            embedded to this dimension. If None, then node and neighbour features are used as passed
        p_dropout (float=0.2): dropout fraction 

    Attributes:

        node_attribute_dim (int)
        edge_attribute_dim (Optional[int]=None)
        embedding_dim (Optional[int]=None)
        p_dropout (float=0.2)
        embed (bool): if embedding_dim is not None, this is set to True, else False
        dim (int): equals node_attribute_dim if embedding_dim is None else embedding_dim

        (Layers)
        node_embedding_layer (nn.Linear): if embedding_dim is not None, this linear layer is created.
            input_dim = node_attribute_dim
            output_dim = embedding_dim
        neighbour_embedding_layer (nn.Linear): if embedding_dim is not None, this linear layer is created.
            input_dim = node_attribute_dim (+ edge_attribute_dim if edge_attribute_dim is not None)
            output_dim = embedding_dim
        alignment_layer (nn.Linear): returns a learned edge weight between node and neighbours
            input_dim = 2 * dim
            output_dim = 1
        context_layer (nn.Linear): learns an update message between a node and its neighbour
            input_dim = 2 * dim
            output_dim = dim
        readout_layer (nn.GRUCell): learns how to update a node embedding with a message
            input_dim = dim
            hidden_layer_size = dim
            output_dim = dim
        dropout (nn.Dropout): dropout layer with fraction = p_dropout

        (set during forward pass)
        neighbours (torch.tensor): 
            tensor containing ordered neighbours attributes neighbour_ij for i in nodes for j in node_neighbours
            where i is ordered from 0 -> num_nodes and j from 0, ..., max(neighbour)
            neighbour attributes have dimension of node_attribute_dim (+ edge_attribute_dim if edge_attr is not None)
            set prior to embedding
        atom_batch_index (torch.tensor.long()): 
            tensor of len(neighbours) detailing to which node the neighbour is a neighbour to
        neighbour_indices (torch.tensor):
            tensor of len(neighbours) detailing the index of neigbour in neighbours
            can be used with batch index to index attentions to node pair
        neighbour_counts (torch.tensor):
            tensor of len(nodes) detailing the neighbour count for that node
        attentions (torch.tensor):
            tensor of len(neighbours), containing the attention between the each node in neighbours
            and the source node in atom_batch_index
        node_embeddings (torch.tensor):
            a tensor of len(nodes) containing the updated node embedding after a forward pass

    Methods:

        forward (x, edge_index, edge_attr):

            nodes = x
            neighbours, atom_batch_index, neighbour_indices, neighbour_counts = self.get_neighbour_attributes(x, edge_index, edge_attr)

            self.neighbours = neighbours
            self.atom_batch_index = atom_batch_index
            self.neighbour_indices = neighbour_indices
            self.neighbour_counts = neighbour_counts

            if self.embed:
                nodes = F.leaky_relu(self.atom_embedding_layer(nodes))
                neighbours = F.leaky_relu(self.neighbour_embedding_layer(neighbours))
            
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

            return readout

        get_neighbour_attributes(x, edge_index, edge_attr):

            reverse_order = torch.concat([edge_index[1, :], edge_index[0, :]], axis=0) # 2 by n_edges
            all_node_pairs = torch.concat([edge_index, reverse_order], axis=1) # 2 by 2 * n_edges
            all_edge_attr = torch.concat([edge_attr, edge_attr]) # 2 by 2 * n_edges
            all_node_attr = x[all_node_pairs[1, :]] # 2 * n_edges
            all_attr = torch.concat([all_node_attr, all_edge_attr], axis=1) 
            argsort = torch.argsort(all_node_pairs[0, :])
            neighbour_attributes = all_attr[argsort]
            atom_batch_index = all_node_pairs[0, :]
            _, neighbour_counts = torch.unique(atom_batch_index, return_counts=True)

            return neighbour_attributes, atom_batch_index, neighbour_counts

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
            self.node_embedding_layer = nn.Linear(self.node_attribute_dim, self.embedding_dim)
            self.neighbour_embedding_layer = \
                nn.Linear(self.node_attribute_dim + self.edge_attribute_dim, self.embedding_dim) \
                if self.edge_attribute_dim is not None \
                else nn.Linear(self.node_attribute_dim, self.embedding_dim)
            self.dim = self.embedding_dim
        else:
            self.embed = False
            self.dim = self.node_attribute_dim
        
        self.alignment_layer = nn.Linear(2*self.dim, 1)
        self.context_layer = nn.Linear(self.dim, self.dim)
        self.readout_layer = nn.GRUCell(self.dim, self.dim)

        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None,
        ) -> torch.tensor:

        '''
        Args:

            x (torch.tensor): node feature matrix (n_nodes by n_node_features)
            edge_index (toch.tensor): adjacency matrix in COO format (2 by n_bonds) where [0, i] are source nodes
                                      and [1, i] are target nodes for the ith bond
            edge_attr (torch.tensor): edge feature matrix (n_bonds by n_bond_features)
        '''

        nodes = x
        neighbours, atom_batch_index, neighbour_indices, neighbour_counts = self.get_neighbour_attributes(x, edge_index, edge_attr)

        self.neighbours = neighbours
        self.atom_batch_index = atom_batch_index
        self.neighbour_indices = neighbour_indices
        self.neighbour_counts = neighbour_counts

        if self.embed:
            nodes = F.leaky_relu(self.node_embedding_layer(nodes))
            neighbours = F.leaky_relu(self.neighbour_embedding_layer(neighbours))
        
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

        return readout

    def get_neighbour_attributes(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None,
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:

        '''
        Args:

            x (torch.tensor): node feature matrix (n_nodes by n_node_features)
            edge_index (toch.tensor): adjacency matrix in COO format (2 by n_bonds) where [0, i] are source nodes
                                      and [1, i] are target nodes for the ith bond
            edge_attr (torch.tensor): edge feature matrix (n_bonds by n_bond_features)
        '''

        reverse_order = torch.vstack([edge_index[1, :], edge_index[0, :]]) # 2 by n_edges
        all_node_pairs = torch.concat([edge_index, reverse_order], axis=1) # 2 by 2 * n_edges
        neighbour_node_attr = x[all_node_pairs[1, :]] # 2 * n_edges
        if edge_attr is not None:
            neighbour_edge_attr = torch.concat([edge_attr, edge_attr]) # 2 by 2 * n_edges
            neighbour_attributes = torch.concat([neighbour_node_attr, neighbour_edge_attr], axis=1) 
        else:
            neighbour_attributes = neighbour_node_attr
        argsort = torch.argsort(all_node_pairs[0, :], stable=True)
        neighbour_attributes = neighbour_attributes[argsort]
        atom_batch_index = all_node_pairs[0, argsort]
        _, neighbour_counts = torch.unique(atom_batch_index, return_counts=True)
        neighbour_counts = neighbour_counts.long()
        neighbour_indices = all_node_pairs[1, argsort]

        return neighbour_attributes, atom_batch_index, neighbour_indices, neighbour_counts