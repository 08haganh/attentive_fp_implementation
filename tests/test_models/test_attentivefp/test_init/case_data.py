from typing import Union

from .....models import AttentiveFP

class CaseData:

    def __init__(
        self,
        model: AttentiveFP,
        node_attribute_dim: int,
        edge_attribute_dim: Union[int, None],
        embedding_dim: int,
        p_dropout: float,
        output_dim: int,
        n_graph_layers: int,
        n_supergraph_layers: int,
        n_linear_layers: int,
        n_graph_descriptors: Union[int, None],
        graph_descriptors_dims: Union[int, None]
        ) -> None:

        self.model = model
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout
        self.output_dim = output_dim
        self.n_graph_layers = n_graph_layers
        self.n_supergraph_layers = n_supergraph_layers
        self.n_linear_layers= n_linear_layers
        self.n_graph_descriptors = n_graph_descriptors
        self.graph_descriptors_dims = graph_descriptors_dims