from ....modules.node_layer import NodeLayer
from .case_data import CaseData

import torch
import numpy as np

def test_get_neighbour_node_attributes(case_data):
    node_layer = NodeLayer(
        node_attribute_dim=3,
        edge_attribute_dim=3,
        embedding_dim=None,
        p_dropout=0.2
    )

    neighbour_node_attributes, atom_batch_index, neighbour_counts = \
        node_layer.get_neighbour_node_attributes(
            case_data.x,
            case_data.edge_index,
    )

    assert torch.all(neighbour_node_attributes == case_data.neighbour_node_attributes)
    assert torch.all(atom_batch_index == case_data.atom_batch_index)
    assert np.all(neighbour_counts == case_data.neighbour_counts)

def test_get_neighbour_edge_attributes(case_data):
    node_layer = NodeLayer(
        node_attribute_dim=3,
        edge_attribute_dim=3,
        embedding_dim=None,
        p_dropout=0.2
    )

    neighbour_edge_attributes, atom_batch_index, neighbour_counts = \
        node_layer.get_neighbour_edge_attributes(
            case_data.x,
            case_data.edge_index,
            case_data.edge_attr
    )

    assert torch.all(neighbour_edge_attributes == case_data.neighbour_edge_attributes)
    assert torch.all(atom_batch_index == case_data.atom_batch_index)
    assert np.all(neighbour_counts == case_data.neighbour_counts)

def test_get_neighbour_all_attributes(case_data):
    node_layer = NodeLayer(
        node_attribute_dim=3,
        edge_attribute_dim=3,
        embedding_dim=None,
        p_dropout=0.2
    )

    neighbour_all_attributes, atom_batch_index, neighbour_counts = \
        node_layer.get_neighbour_all_attributes(
            case_data.x,
            case_data.edge_index,
            case_data.edge_attr
    )

    assert torch.all(neighbour_all_attributes == case_data.neighbour_all_attributes)
    assert torch.all(atom_batch_index == case_data.atom_batch_index)
    assert np.all(neighbour_counts == case_data.neighbour_counts)

