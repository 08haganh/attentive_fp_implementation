from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng

from ..modules import AFPGraphConvolution, AFPSuperGraphConvolution

class AttentiveFP(nng.MessagePassing):

    '''
    AttentiveFP implentation, allowing the use of graph level descriptors

    Attributes:

        (init args)
        node_attribute_dim: (int)
            the number of node features
        edge_attribute_dim: (Optional[int]=None)
            the number of edge features
        embedding_dim: (int)
            the dimension to embed the node features, neighbour features,
            and graph_descriptors to
            i.e. the number of features of the graph embedding
        p_dropout: (float)
            the dropout probability for the dropout layers
        output_dim: (int)
            the dimensionality of the outputs
        n_graph_layers: (int)
            the number of graph convolutions to be completed
        n_supergraph_convolutions: (int)
            the number of convolutions to be completed on the supergraph
        n_linear_layers: (int)
            the number of linear layers
        n_graph_descriptors: (Optional[int]=None)
            the number of graph descriptors to be expected
        graph_descriptors_dims: (Optional[List[int]]=None)
            the number of features in each of the graph descriptors
            dim=(n_graph_descriptors)

        (layers)
        graph_layers: (nn.ModuleList[AFPGraphConvolution])
            the layers for completing graph convolutions
            of len=n_graph_layers
        supergraph_layers: (nn.ModuleList[AFPSuperGraphConvolution])
            the layers for completing supergraph convolutions
            of len=n_supergraph_layers
        linear_layers: (nn.ModuleList[nn.Linear])
            the linear layers of the network of len=n_linear_layers
        dropout: (nn.Dropout)
            dropout layer
        graph_descriptor_embedding_layers: 
            (Union[nn.ModuleList[nn.Linear], None])
            equals None if self.n_graph_descriptors is None else
            the linear layers for embedding the graph descriptors 
            of len=n_graph_descriptors
    '''

    def __init__(
        self,
        node_attribute_dim: int,
        edge_attribute_dim: Optional[int]=None,
        embedding_dim: int=256,
        p_dropout: float=0,
        output_dim: int=1,
        n_graph_layers: int=1,
        n_supergraph_layers: int=1,
        n_linear_layers: int=1,
        n_graph_descriptors: Optional[int]=None,
        graph_descriptors_dims: Optional[List[int]]=None
        ) -> None:

        '''
        Initialises an AttentiveFP instance

        Args:

            node_attribute_dim: (int)
                the number of node features
            edge_attribute_dim: (Optional[int]=None)
                the number of edge features
            embedding_dim: (int)
                the dimension to embed the node features, neighbour features,
                and graph_descriptors to
                i.e. the number of features of the graph embedding
            p_dropout: (float)
                the dropout probability for the dropout layers
            output_dim: (int)
                the dimensionality of the outputs
            n_graph_layers: (int)
                the number of graph convolutions to be completed
            n_supergraph_convolutions: (int)
                the number of convolutions to be completed on the supergraph
            n_linear_layers: (int)
                the number of linear layers
            n_graph_descriptors: (Optional[int]=None)
                the number of graph descriptors to be expected
            graph_descriptors_dims: (Optional[List[int]]=None)
                the number of features in each of the graph descriptors
                dim=(n_graph_descriptors)
        '''

        super(AttentiveFP, self).__init__()

        # init args
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout
        self.output_dim = output_dim
        self.n_graph_layers = n_graph_layers
        self.n_supergraph_layers = n_supergraph_layers
        self.n_linear_layers = n_linear_layers
        self.n_graph_descriptors = n_graph_descriptors
        self.graph_descriptors_dims = graph_descriptors_dims

        # make layers
        ### graph layers
        self.graph_layers = nn.ModuleList([
            AFPGraphConvolution(
                node_attribute_dim=self.node_attribute_dim, 
                edge_attribute_dim=self.edge_attribute_dim,
                embedding_dim=self.embedding_dim, 
                p_dropout=p_dropout
                )])

        self.graph_layers.extend([
            AFPGraphConvolution(
                node_attribute_dim=embedding_dim, 
                p_dropout=p_dropout)
            for _ in range(n_graph_layers-1)
         ])
        
        ### supergraph layers
        self.supergraph_layers = nn.ModuleList([
            AFPSuperGraphConvolution(
                dim=embedding_dim,
                p_dropout=p_dropout)
            for _ in range(n_supergraph_layers)
         ])

        ### linear layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(n_linear_layers - 1)
         ])
        self.linear_layers.append(nn.Linear(embedding_dim, output_dim))

        ### dropout
        self.dropout = nn.Dropout(p=p_dropout)

        ### graph descriptor embedding layers
        if self.n_graph_descriptors is not None:
            self.graph_descriptor_embedding_layers = nn.ModuleList([
                nn.Linear(dim, embedding_dim)
                for dim in graph_descriptors_dims
            ])
        else:
            self.graph_descriptor_embedding_layers = None

    def forward(
        self,
        x: torch.tensor, 
        edge_index: torch.tensor,
        edge_attr: Optional[torch.tensor]=None,
        batch_index: Optional[torch.tensor]=None,
        graph_x: Optional[List]=None
        ) -> torch.tensor:

        '''
        a forward pass through the model.

        Args:

            x: (torch.tensor)
                the node feature matrix
            edge_index: (torch.tensor)
                the adjacency matrix in COO format
            edge_attr: (Optional[torch.tensor]=None)   
                the edge feature matrix
            batch_index: (Optional[torch.tensor]=None)
                the index of the graph to which each feature in x 
                belongs. If this is None, it is assumed that all 
                nodes belong to the same graph
            graph_x: (Optional[List[List[torch.tensor]]]=None)
                graph level descriptors for each graph in the batch
                of dim=(n_graphs, n_graph_descriptors, varied)
        '''

        # embed each of the graph descriptors
        # and store in torch.tensor of 
        # dims=(n_graphs, n_graph_descriptors, self.embedding_dim)
        if graph_x is not None:
            embedded_graph_x = torch.empty(size=[len(graph_x), self.n_graph_descriptors, self.embedding_dim], dtype=torch.float32)
            for j, mod in enumerate(self.graph_descriptor_embedding_layers):
                graph_descs = torch.vstack([graph_x[i][j] for i in range(len(graph_x))])
                embedded_graph_x[:, j, :] = mod(self.dropout(graph_descs))
        else:
            embedded_graph_x = None
        

        # graph convolutions
        x = self.graph_layers[0](x, edge_index, edge_attr)
        for mod in self.graph_layers[1:]:
            x = mod(x, edge_index)

        # supergraph convolutions
        graph_embedding = self.supergraph_layers[0](x, batch_index, graph_x=embedded_graph_x)
        for mod in self.supergraph_layers[1:]:
            graph_embedding = mod(x, batch_index, graph_embedding, embedded_graph_x)

        # Linear layers
        for mod in self.linear_layers[:-1]:
            graph_embedding = F.leaky_relu(mod(self.dropout(graph_embedding)))

        return self.linear_layers[-1](self.dropout(graph_embedding))
