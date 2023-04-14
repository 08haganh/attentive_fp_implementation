from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as nng
from torch_geometric.utils import softmax

class AFPConvolution(nn.Module):

    '''
    Base class for AttentiveFP convolution layers, defining the general convolution 
    mechanism for updating node embeddings for both the graph
    and supervirtual graph representations of the graph.
     
    Inherited classes are required to write their own forward() 
    and _get_neighbour_attributes() methods

    Attributes:

        dim (torch.tensor):
            the expected dimension of the nodes and neighbours
        p_dropout (float):
            the dropout probability for dropout layers
        alignment_layer(nn.Linear):
            a linear layer for computing alignment scores between node-neighbour pairs
            in_features=(2 * dim)
            out_features=(1)
        context_layer (nn.Linear):
            a linear layer for computing neighbour messages to nodes
            in_features=(dim)
            out_features=(dim)
        readout_layer (nn.GRUCell):
            a GRUCell for updating node embeddings with their context / messages
            input_size=(dim)
            hidden_size=(dim)
        attentions (Union[None, torch.tensor]):
            set during the forward pass, the attention coefficients for 
            each node-neighbour pair
        node_index (Union[None, torch.tensor]):
            set during forward pass, the index of the node for each 
            node-neighbour pair in attentions
        neighbour_index (Union[None, torch.tensor]):
            set during forward pass, the index of the neighbour 
            for each node-neighbour pair in attentions

    Methods:

        forward()
            a forward pass through the layer
        _get_neighbour_attributes()
            returns variables necessary to complete the convolution
        _update_node_embeddings()
            updates the node_embeddings
        _expand_and_join()
            expands the node_embeddings and joins them
            to their corresponding neighbour features
    '''

    def __init__(
        self,
        dim: int,
        p_dropout: float=0
        ) -> None:
        
        '''
        initialises the AFPConvolution

        Args:

            dim (int):
                the expected dimension of the nodes and neighbours
            p_dropout (float):
                the dropout probability for dropout layers
        '''

        super(AFPConvolution, self).__init__()

        self.dim = dim
        self.p_dropout = p_dropout

        # layers
        self.alignment_layer = nn.Linear(2 * dim, 1)
        self.context_layer = nn.Linear(dim, dim)
        self.readout_layer = nn.GRUCell(dim, dim) 
        self.dropout = nn.Dropout(p_dropout)

        # attentions
        self.node_index = None
        self.neighbour_index = None
        self.attentions = None

    def forward(self) -> torch.tensor:

        '''
        a forward pass through the layer. 
        
        Returns:
            
            readout (torch.tensor):
                updated node embeddings
        '''

        raise NotImplementedError
    
    def _get_neighbour_attributes(self):

        '''
        Returns:

            neighbour_attributes: torch.tensor
                a torch.tensor of the neighbour features for all node-neighbour pairs 
                in the graph. The ordering should be in [x[j] for i in range(n_nodes) for j in node[i].neighbour_indices]
                i.e. i is in ascending order, but j is in order of occurence in edge_index
            batch_index: torch.tensor
                a torch.tensor of the index in x of the node for each node-neighbour pair
                in neighbour_attributes
            neighbour_indices: torch.tensor
                a torch.tensor of the index in x of the neighbour in each node-neighbour pair
                in neighbour_attributes
            neighbour_counts (List[int]):
                a list of the number of neighbours for each node in x
                [node[i].num_neighbours for i in range(n_nodes)]
        '''

        raise NotImplementedError
    
    def _update_node_embedding(
        self, 
        node_embeddings: torch.tensor,
        neighbour_attributes: torch.tensor, 
        neighbour_counts: List[int],
        batch_index: torch.tensor,
        ) -> torch.tensor:

        '''
        updates node embeddings using an attention based graph convolution
        operation. 

        Args:

            node_embeddings (torch.tensor):
                the node embeddings to be updated
            neighbour_attributes (torch.tensor):
                the neighbour features for each node-neighbour pair in the graph
                arranged in ascending order of nodes
                concat([node_0_neighbours, ..., node_n_neighbours])
            neighbour_counts (List[int]):
                the number of neighbours for each node
            batch_index (torch.tensor):
                the node to which each node-neighbour pair
                in neighbour_attributes belongs

        Returns:

            readout (torch.tensor):
                updated node_embeddings
        '''

        joint_attributes = self._expand_and_join(
            node_embeddings,
            neighbour_attributes,
            neighbour_counts
            )
        
        alignment = F.leaky_relu(self.alignment_layer(self.dropout(joint_attributes)))
        attentions = softmax(alignment, batch_index)
        contexts = F.elu(
            nng.global_add_pool(
                torch.mul(attentions, self.context_layer(self.dropout(neighbour_attributes))), 
                batch_index
                )
            )

        readout = F.relu(self.readout_layer(contexts, node_embeddings))
        self.attentions = attentions

        return readout
    
    def _expand_and_join(
        self,
        node_embeddings: torch.tensor,
        neighbour_attributes: torch.tensor,
        neighbour_counts: torch.tensor,
        ) -> torch.tensor:

        '''
        expands node_embeddings and joins them to their 
        neighbour_attributes

        Args:

            node_embeddings (torch.tensor):
                the node embeddings to be updated
            neighbour_attributes (torch.tensor):
                the neighbour features for each node-neighbour pair in the graph
                arranged in ascending order of nodes
                concat([node_0_neighbours, ..., node_n_neighbours])
            neighbour_counts (List[int]):
                the number of neighbours for each node

        Returns:

            joint_attributes (torch.tensor):
                concatenated features for each node-neighbour pair
        '''

        expanded = torch.concat([node_embeddings[i].expand(index, node_embeddings.shape[1]) for i, index in enumerate(neighbour_counts)])
        joint_attributes = torch.concat([expanded, neighbour_attributes], axis=1)

        return joint_attributes