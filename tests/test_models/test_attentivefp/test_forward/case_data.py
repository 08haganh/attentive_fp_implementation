from typing import Optional, List

from .....models import AttentiveFP
import torch

class CaseData:

    def __init__(
        self,
        model: AttentiveFP,
        x: torch.tensor,
        edge_index: torch.tensor,
        output_dim: torch.Size,
        edge_attr: Optional[torch.tensor]=None,
        batch_index: Optional[torch.tensor]=None,
        graph_x: Optional[torch.tensor]=None,
        ) -> None:

        self.model = model
        self.x = x
        self.edge_index = edge_index
        self.output_dim = output_dim
        self.edge_attr = edge_attr
        self.batch_index = batch_index
        self.graph_x = graph_x