from typing import Union

from ....modules import AFPSuperGraphConvolution
import torch

class CaseData:

    def __init__(
        self,
        layer: AFPSuperGraphConvolution,
        x: torch.tensor,
        batch_index: Union[torch.tensor, None],
        graph_embeddings: Union[torch.tensor, None],
        graph_x: Union[torch.tensor, None],
        readout_shape: torch.Size,
        expected_attentions: torch.tensor,
        node_index: torch.tensor,
        neighbour_index: torch.tensor
        ) -> None:

        self.layer = layer
        self.x = x
        self.batch_index = batch_index
        self.graph_embeddings = graph_embeddings
        self.graph_x = graph_x
        self.readout_shape = readout_shape
        self.expected_attentions = expected_attentions
        self.node_index = node_index
        self.neighbour_index = neighbour_index
