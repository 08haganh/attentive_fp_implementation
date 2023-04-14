from .case_data import CaseData
import torch

def test_forward(case_data: CaseData):

    readout = case_data.layer(
        case_data.x,
        case_data.edge_index,
        case_data.edge_attr
        )
    
    assert readout.shape == case_data.readout_shape
    assert torch.allclose(case_data.layer.attentions, case_data.expected_attentions, atol=0.01)
    assert torch.all(case_data.layer.node_index == case_data.node_index)
    assert torch.all(case_data.layer.neighbour_index == case_data.neighbour_index)