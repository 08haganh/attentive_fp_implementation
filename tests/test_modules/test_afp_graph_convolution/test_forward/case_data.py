from typing import Tuple

from ....modules import AFPGraphConvolution
import torch

class CaseData:

    def __init__(
        self,
        layer: AFPGraphConvolution,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attr: torch.tensor,
        readout_shape: Tuple[int, int],
        expected_attentions: torch.tensor,
        node_index: torch.tensor,
        neighbour_index: torch.tensor
        ) -> None:

        self.layer = layer
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.readout_shape = readout_shape
        self.expected_attentions = expected_attentions
        self.node_index = node_index
        self.neighbour_index = neighbour_index