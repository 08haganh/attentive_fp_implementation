from ....modules.node_layer import NodeLayer
from .case_data import CaseData

import torch

def test_get_neighbours(
    case_data: CaseData
    ) -> None:
    

    node_layer = NodeLayer(
        node_attribute_dim=case_data.x.shape[1],
        edge_attribute_dim=case_data.edge_attr.shape[1],
        embedding_dim=20,
        p_dropout=0
    )

    _ = node_layer(
        case_data.x, 
        case_data.edge_index, 
        case_data.edge_attr
        )
    
    assert torch.all(node_layer.neighbours == case_data.neighbour_attributes)
    assert torch.all(node_layer.atom_batch_index == case_data.atom_batch_index)
    assert torch.all(node_layer.neighbour_indices == case_data.neighbour_indices)
    assert torch.all(node_layer.neighbour_counts == case_data.neighbour_counts)

def test_get_neighbours_no_edges(
    case_data_no_edge_attr: CaseData
    ) -> None:
    

    node_layer = NodeLayer(
        node_attribute_dim=case_data_no_edge_attr.x.shape[1],
        edge_attribute_dim=None,
        embedding_dim=20,
        p_dropout=0
    )

    _ = node_layer(
        case_data_no_edge_attr.x, 
        case_data_no_edge_attr.edge_index, 
        )
    
    assert torch.all(node_layer.neighbours == case_data_no_edge_attr.neighbour_attributes)
    assert torch.all(node_layer.atom_batch_index == case_data_no_edge_attr.atom_batch_index)
    assert torch.all(node_layer.neighbour_indices == case_data_no_edge_attr.neighbour_indices)
    assert torch.all(node_layer.neighbour_counts == case_data_no_edge_attr.neighbour_counts)
