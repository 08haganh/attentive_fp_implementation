import pytest
from .case_data import CaseData

import torch
import numpy as np

@pytest.fixture
def node_attributes():
    return torch.tensor(np.array([
        [0, 0 ,0], # graph 1
        [0, 0, 1],
        [0, 1, 0],
        [1, 0 ,0],
        [1, 1, 1],

        [2, 0, 0], # graph 2
        [0, 2, 0],
        [0, 0, 2]
    ])).float()

@pytest.fixture
def edge_index():
    return torch.tensor(np.array([
        [0, 1, 2, 2, 3, 5, 6, 7],
        [1, 2, 0, 3, 4, 6, 7, 5]
    ]))

@pytest.fixture
def edge_attributes():
    return torch.tensor(np.array([
        [1, 2, 3], # graph 1
        [1, 2, 4],
        [1, 2, 5],
        [1, 2, 6],
        [1, 2, 7],

        [5, 5, 5], # graph 2
        [5, 6, 5],
        [6, 5, 5]
    ]))

@pytest.fixture
def batch_index():
    return torch.tensor([
        0, 0, 0, 0, 0, 1, 1, 1
    ]).long()

@pytest.fixture
def graph_nodes(node_attributes):
    return torch.concatenate([
        torch.sum(node_attributes[:5], axis=0).reshape(1, 3),
        torch.sum(node_attributes[5:], axis=0).reshape(1, 3)
        ], axis=0)

@pytest.fixture
def expanded(batch_index, graph_nodes):
    _, atom_counts = torch.unique(batch_index, return_counts=True)
    return torch.concat([graph_nodes[i].expand(ac, graph_nodes.shape[1]) for i, ac in enumerate(atom_counts)], axis=0)

@pytest.fixture
def joint_attributes(expanded, node_attributes):
    return torch.concat([expanded, node_attributes], axis=1)

@pytest.fixture
def case_data(node_attributes, batch_index, graph_nodes, joint_attributes):
    return CaseData(node_attributes, batch_index, graph_nodes, joint_attributes)
    