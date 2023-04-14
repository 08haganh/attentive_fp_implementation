from .case_data import CaseData
import torch

def test_get_neighbours(case_data: CaseData):

    neighbour_attributes, batch_index, neighbour_index, neighbour_counts = \
        case_data.layer._get_neighbour_attributes(
            case_data.x, 
            case_data.batch_index, 
            case_data.graph_x
        )
    
    assert torch.all(batch_index == case_data.return_batch_index)
    assert torch.all(neighbour_index == case_data.neighbour_index)
    assert neighbour_counts == case_data.neighbour_counts
    assert torch.all(neighbour_attributes == case_data.neighbour_attributes)
