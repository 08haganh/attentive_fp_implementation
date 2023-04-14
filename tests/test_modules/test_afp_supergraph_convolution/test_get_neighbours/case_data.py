from typing import Union, List

from ....modules import AFPSuperGraphConvolution
import torch

class CaseData:

    def __init__(
        self,
        layer: AFPSuperGraphConvolution,
        x: torch.tensor,
        batch_index: torch.tensor,
        graph_x : Union[torch.tensor, None],
        neighbour_attributes: torch.tensor,
        return_batch_index: torch.tensor,
        neighbour_index: torch.tensor,
        neighbour_counts: List[int]
        ) -> None:

        self.layer = layer
        self.x = x
        self.batch_index = batch_index
        self.graph_x = graph_x
        self.neighbour_attributes = neighbour_attributes
        self.return_batch_index = return_batch_index
        self.neighbour_index = neighbour_index
        self.neighbour_counts = neighbour_counts