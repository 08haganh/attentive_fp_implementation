from typing import List
from .....modules import AFPGraphConvolution
import torch

class CaseData:

    def __init__(
        self,
        layer: AFPGraphConvolution,
        x: torch.tensor,
        edge_index: torch.tensor,
        edge_attributes: torch.tensor,
        neighbour_attributes: torch.tensor,
        node_index: torch.tensor,
        neighbour_index: torch.tensor,
        neighbour_counts: List[int],
        ) -> None:

        self.layer = layer
        self.x = x
        self.edge_index = edge_index
        self.edge_attributes = edge_attributes
        self.neighbour_attributes = neighbour_attributes
        self.node_index = node_index
        self.neighbour_index = neighbour_index
        self.neighbour_counts = neighbour_counts