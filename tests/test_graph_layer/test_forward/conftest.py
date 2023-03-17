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
def graph_embeddings(node_attributes):
    return torch.concatenate([torch.sum(node_attributes[:5], axis=0),
         torch.sum(node_attributes[5:], axis=0)
        ], axis=0)

@pytest.fixture
def case_data():
    return CaseData()
    