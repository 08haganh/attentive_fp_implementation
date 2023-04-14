from typing import Union

from .....modules import AFPGraphConvolution

class CaseData:

    def __init__(
        self,
        layer: AFPGraphConvolution,
        node_attribute_dim: int,
        edge_attribute_dim: Union[int, None],
        embedding_dim: Union[int, None],
        p_dropout: float,
        ) -> None:

        self.layer = layer
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout