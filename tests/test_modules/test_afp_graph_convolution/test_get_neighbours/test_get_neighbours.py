from .case_data import CaseData
import torch
def test_get_neighbours_with_edge_attributes(case_data: CaseData):
    
    neighbour_attributes, node_index, neighbour_index, neighbour_counts = \
        case_data.layer._get_neighbour_attributes(
            case_data.x, 
            case_data.edge_index, 
            case_data.edge_attributes
        )

    assert torch.all(node_index == case_data.node_index)
    assert torch.all(neighbour_index == case_data.neighbour_index)
    assert torch.all(neighbour_attributes == case_data.neighbour_attributes)
    assert neighbour_counts == case_data.neighbour_counts