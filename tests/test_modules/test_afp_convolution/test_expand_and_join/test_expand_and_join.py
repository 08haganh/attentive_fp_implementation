from .case_data import CaseData
import torch

def test_expand_and_join(case_data:  CaseData):

    joint_attributes = case_data.layer._expand_and_join(
        case_data.node_embeddings,
        case_data.neighbour_attributes,
        case_data.neighbour_counts
    )

    assert torch.all(joint_attributes == case_data.joint_attributes)