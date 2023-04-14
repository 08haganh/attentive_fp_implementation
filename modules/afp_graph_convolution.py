from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .afp_convolution import AFPConvolution

class AFPGraphConvolution(AFPConvolution):

    '''
    An AttentiveFP layer for updating node embeddings

    Attributes:

        node_attribute_dim (int):
            the number of node features
        edge_attribute_dim (Optional[int]=None):
            the number of edge features
        embedding_dim (Optional[int]=None):
            the dimension to embed the node and neighbour features to
            if None, no embedding is completed
        node_embedding_layer (Union[None, nn.Linear]):
            None if embedding_dim is None
            else linear layer to embed the node features. 
                in_features = node_attribute_dim
                out_features = embedding_dim
        neighbour_embedding_layer (Union[None, nn.Linear]):
            None if embedding_dim is None
            else linear layer to embed the node features. 
                in_features = node_attribute_dim (+ edge_attribute_dim if edge_attribute_dim is not None)
                out_features = embedding_dim 
    '''

    def __init__(
        self,
        node_attribute_dim: int,
        edge_attribute_dim: Optional[int]=None,
        embedding_dim: Optional[int]=None,
        p_dropout: float=0
        ) -> None:
        
        '''
        Args:

            node_attribute_dim (int):
                the number of node features
            edge_attribute_dim (Optional[int]=None):
                the number of edge features
            embedding_dim (Optional[int]=None):
                the dimension to embed the node and neighbour features to
                if None, no embedding is completed
            p_dropout (float):
                the dropout probability for dropout layers
        '''

        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        dim = node_attribute_dim if embedding_dim is None else embedding_dim

        super(AFPGraphConvolution, self).__init__(
            dim=dim,
            p_dropout=p_dropout
        )

        # layers
        if embedding_dim is not None:
            self.node_embedding_layer = nn.Linear(node_attribute_dim, embedding_dim)
            self.neighbour_embedding_layer = \
                nn.Linear(node_attribute_dim + edge_attribute_dim, embedding_dim) \
                if edge_attribute_dim is not None \
                else nn.Linear(node_attribute_dim, embedding_dim)
        else:
            self.node_embedding_layer = None
            self.neighbour_embedding_layer = None

    def forward(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None
        ) -> torch.tensor:

        '''
        a forward pass through the layer. 

        Args:

            x (torch.tensor):
                the node attributes
            edge_index (torch.tensor):
                the adjacency matric in COO format
            edge_attr (Optional[torch.tensor]=None):
                the edge attributes
        
        Returns:
            
            node_embeddings (torch.tensor)
                the updated node_embeddings
        '''

        node_embeddings = x
        neighbour_attributes, batch_index, neighbour_index, neighbour_counts = self._get_neighbour_attributes(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr
            )

        if self.node_embedding_layer is not None:
            node_embeddings = F.leaky_relu(self.node_embedding_layer(node_embeddings))
            neighbour_attributes = F.leaky_relu(self.neighbour_embedding_layer(neighbour_attributes))

        readout = self._update_node_embedding(
            node_embeddings, 
            neighbour_attributes, 
            neighbour_counts,
            batch_index
            )
        
        self.node_index = batch_index
        self.neighbour_index = neighbour_index

        return readout
    
    def _get_neighbour_attributes(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None
        ) -> Tuple[torch.tensor]:

        '''
        Args:

            x (torch.tensor):
                the node attributes
            edge_index (torch.tensor):
                the adjacency matric in COO format
            edge_attr (Optional[torch.tensor]=None):
                the edge attributes

        Returns:

            neighbour_attributes: torch.tensor
                a torch.tensor of the neighbour features for all node-neighbour pairs 
                in the graph. The neighbour attributes are the neighbours node features and 
                the node-neighbour edge attribute if edge_attr is not None else
                only the node features
                The ordering should be in [x[j] for i in range(n_nodes) for j in node[i].neighbour_indices]
                i.e. i is in ascending order, but j is in order of occurence in edge_index
            batch_index: torch.tensor
                a torch.tensor of the index in x of the node for each node-neighbour pair
                in neighbour_attributes
            neighbour_index: torch.tensor
                a torch.tensor of the index in x of the neighbour in each node-neighbour pair
                in neighbour_attributes
            neighbour_counts: List[int]
                a list of the number of neighbours for each node in x
                [node[i].num_neighbours for i in range(n_nodes)]
        '''

        device = x.device

        reverse_order = torch.vstack([edge_index[1, :], edge_index[0, :]]) # 2 by n_edges
        all_node_pairs = torch.concat([edge_index, reverse_order], axis=1) # 2 by 2 * n_edges
        neighbour_node_attr = x[all_node_pairs[1, :]] # 2 * n_edges
        if edge_attr is not None:
            neighbour_edge_attr = torch.concat([edge_attr, edge_attr]) # 2 by 2 * n_edges
            neighbour_attributes = torch.concat([neighbour_node_attr, neighbour_edge_attr], axis=1) 
        else:
            neighbour_attributes = neighbour_node_attr
        argsort = torch.argsort(all_node_pairs[0, :], stable=True)
        neighbour_attributes = neighbour_attributes[argsort].to(device)
        batch_index = all_node_pairs[0, argsort].long().to(device)
        _, neighbour_counts = torch.unique(batch_index, return_counts=True)
        neighbour_counts = neighbour_counts.long().detach().numpy().tolist()
        neighbour_index = all_node_pairs[1, argsort].long().to(device)

        return (
            neighbour_attributes,
            batch_index,
            neighbour_index,
            neighbour_counts
            )
    