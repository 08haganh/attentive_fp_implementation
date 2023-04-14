from typing import Tuple, Optional, List

import torch
import torch_geometric.nn as nng

from .afp_convolution import AFPConvolution

class AFPSuperGraphConvolution(AFPConvolution):

    def __init__(
        self,
        dim: int,
        p_dropout: float=0
        ) -> None:

        super(AFPSuperGraphConvolution, self).__init__(
            dim=dim,
            p_dropout=p_dropout
        )

    def forward(
        self,
        x: torch.tensor,
        batch_index: Optional[torch.tensor]=None,
        graph_embeddings: Optional[torch.tensor]=None,
        graph_x: Optional[torch.tensor]=None,
        ) -> torch.tensor:

        '''
        a forward pass through the layer. 

        Args:

            x (torch.tensor):
                the node embeddings of the graph
            batch_index (Optional[torch.tensor]=None):
                the indices of the graphs to which each node
                in x belongs. If this is None, it is assumed 
                all nodes belong to a single graph
            graph_embeddings (Optional[torch.tensor]):
                the graph_embedding for the graph. If this is None,
                a graph_embedding is calculated as the sum of their 
                corresponding node embeddings
            graph_x (Optional[torch.tensor]=None):
                optional graph level features to incorporate into 
                the supergraph. Expected to have 
                dim=(n_graphs, n_graph_descriptors, x.shape[1])
        
        Returns:
            
            readout (torch.tensor):
                updated graph embeddings
        '''

        # check if have batch_index
        if batch_index is None:
            batch_index = torch.zeros(size=[x.shape[0]]).long()

        # make graph_embedding
        if graph_embeddings is None:
            graph_embeddings = nng.global_add_pool(x, batch_index)

        neighbour_attributes, batch_index, neighbour_index, neighbour_counts = self._get_neighbour_attributes(
            x, 
            batch_index, 
            graph_x
            )
        
        print(neighbour_attributes)
        
        readout = self._update_node_embedding(
            graph_embeddings, 
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
        batch_index: torch.tensor,
        graph_x: Optional[torch.tensor]=None, 
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, List[int]]:

        '''
        Args:

            x (torch.tensor):
                the node embeddings of the graph
            batch_index (Optional[torch.tensor]=None):
                the indices of the graphs to which each node
                in x belongs.
            graph_x (Optional[torch.tensor]=None):
                optional graph level features to incorporate into 
                the supergraph. Expected to have 
                dim=(n_graphs, n_graph_descriptors, x.shape[1])

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
            neighbour_counts: List[int]
                a torch.tensor of the number of neighbours for each node in x
                [node[i].num_neighbours for i in range(n_nodes)]
        '''

        device = x.device

        graph_indices, node_counts = torch.unique(batch_index, return_counts=True) 
        n_graph_descriptors = graph_x.shape[1] if graph_x is not None else 0
        node_counts += n_graph_descriptors
        graph_indices = graph_indices.long().cpu().detach().numpy().tolist()
        neighbour_counts = node_counts.long().cpu().detach().numpy().tolist()
        
        if graph_x is not None:
            neighbour_attributes = torch.concat([
                torch.concat([x[batch_index == ind], graph_x[ind]], axis=0)
                for ind in graph_indices
                ],
                axis=0)
        else:
            neighbour_attributes = x

        batch_index = torch.concat([torch.tensor([i]*node_counts[i]) for i in graph_indices], axis=0).long()
        neighbour_index = torch.tensor([x for x in range(neighbour_attributes.shape[0])]).long()

        return (
            neighbour_attributes, 
            batch_index, 
            neighbour_index, 
            neighbour_counts
            )

    
    