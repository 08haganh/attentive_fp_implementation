from ....modules import AFPConvolution
import torch

class CaseData:

    def __init__(
        self,
        layer: AFPConvolution,
        node_embeddings: torch.tensor,
        neighbour_attributes: torch.tensor,
        neighbour_counts: torch.tensor,
        joint_attributes: torch.tensor
        ) -> None:

        self.layer = layer
        self.node_embeddings = node_embeddings
        self.neighbour_attributes = neighbour_attributes
        self.neighbour_counts = neighbour_counts
        self.joint_attributes = joint_attributes