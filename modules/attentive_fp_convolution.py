from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as nng
from torch_geometric.utils import softmax

class AttentiveFPConvolution(nng.MessagePassing):

    '''
    torch_geometric module which updates node embeddings as detailed in X.

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

        (layers)
        node_embedding_layer (nn.Linear, None): linear layer for embedding node features
            input_dim = node_attribute_dim
            output_dim = embedding_dim
        neighbour_embedding_layer (nn.Linear): linear layer for embedding neighbour features
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
            where i is ordered from 0 -> num_nodes and j from the order of occurence of the node-neighbour pair in edge_index
            neighbour attributes have dimension of n_neighbours by node_attribute_dim (+ edge_attribute_dim if edge_attr is not None)
            set prior to embedding
        atom_batch_index (torch.tensor.long()): 
            tensor of len(neighbours) detailing to which node the neighbour is a neighbour to in neighbours
        neighbour_indices (torch.tensor):
            tensor of len(neighbours) detailing the index of neighbour in neighbours
            can be used with atom_batch_index to index attentions to node pair
        neighbour_counts (torch.tensor):
            tensor of len(nodes) detailing the neighbour count for that node
        attentions (torch.tensor):
            tensor of len(neighbours), containing the attention between the each node in neighbours
            and the source node in atom_batch_index
        node_embeddings (torch.tensor):
            a tensor of len(nodes) containing the updated node embedding after a forward pass

    Methods:

        forward (self, x, edge_index, edge_attr) -> torch.tensor
            returns 
                readout: torch.tensor of updated node embeddings
        get_neighbour_attributes(self, x, edge_index, edge_attr)x -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
            returns 
                neighbour_feature: torch.tensor of neighbour features, 
                atom_batch_index; torch.tensor detailing to which node each element in neighbour features is a neighbour to
                neighbour_indices: torch.tensor of the indices of each neighbour
                neighbour_counts: torch.tensor detailing the the number of neighbours of each node i 
    '''
    
    def __init__(
        self,
        node_attribute_dim: int,
        edge_attribute_dim: Optional[int]=None,
        embedding_dim: Optional[int]=None,
        p_dropout: float=0
        ) -> None:
        
        super(AttentiveFPConvolution, self).__init__()

        # args
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout

        # create layers
        if self.embedding_dim is not None:
            self.embed = True
            self.node_embedding_layer = nn.Linear(self.node_attribute_dim, self.embedding_dim)
            self.neighbour_embedding_layer = \
                nn.Linear(self.node_attribute_dim + self.edge_attribute_dim, self.embedding_dim) \
                if self.edge_attribute_dim is not None \
                else nn.Linear(self.node_attribute_dim, self.embedding_dim)
            self.dim = self.embedding_dim
        else:
            self.node_embedding_layer = None
            self.neighbour_embedding_layer = None
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

        returns:

            readout (torch.tensor): updated node embeddings

        Steps:

            1) get neighbours attributes - get a tensor a neighbour features 
            2) embed node and neighbour attributes if self.embed is True
            3) compute alignments
            4) compute attentions
            5) compute contexts
            6) update node embeddings
        '''

        nodes = x
        neighbours, atom_batch_index, neighbour_indices, neighbour_counts = self.get_neighbour_attributes(x, edge_index, edge_attr)

        if self.embed:
            nodes = F.leaky_relu(self.node_embedding_layer(nodes))
            neighbours = F.leaky_relu(self.neighbour_embedding_layer(neighbours))
        
        # complete graph convolution
        expanded = torch.concat([nodes[i].expand(index, nodes.shape[1]) for i, index in enumerate(neighbour_counts)])
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

        self.neighbours = neighbours
        self.atom_batch_index = atom_batch_index
        self.neighbour_indices = neighbour_indices
        self.neighbour_counts = neighbour_counts
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
    
    def __repr__(self) -> str:

        info = {
            'node_attribute_dim':self.node_attribute_dim,
            'edge_attribute_dim':self.edge_attribute_dim,
            'embedding_dim':self.embedding_dim,
            'embed':self.embed,
            'dim':self.dim,
            'p_dropout':self.p_dropout,
            'node_embedding_layer':self.node_embedding_layer,
            'neighbour_embedding_layer':self.neighbour_embedding_layer,
            'alignment_layer':self.alignment_layer,
            'context_layer':self.context_layer,
            'readout_layer':self.readout_layer,
            'dropout':self.dropout
        }

        return f'{info}'