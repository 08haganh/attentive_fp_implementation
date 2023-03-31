import pytest
from .case_data import CaseData

import torch
import numpy as np

@pytest.fixture
def node_attributes():
    return torch.tensor([
        [0, 0 ,0], # 0
        [0, 0, 1], # 1
        [0, 1, 0], # 2
        [1, 0 ,0], # 3 
        [1, 1, 1] # 4
    ], dtype=torch.float
    )

@pytest.fixture
def edge_index():
    return torch.tensor([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 3, 4]
    ], dtype=torch.long
    )

@pytest.fixture
def edge_attributes():
    return torch.tensor([
        [1, 2, 3], # 0-1
        [1, 2, 4], # 1-2
        [1, 2, 5], # 2-0
        [1, 2, 6], # 2-3
        [1, 2, 7] # 3-4
    ], dtype=torch.float
    )

@pytest.fixture
def neighbour_attributes():
    return torch.tensor([
        [0, 0, 1, 1, 2, 3], # 0-1
        [0, 1, 0, 1, 2, 5], # 0-2
        [0, 1, 0, 1, 2, 4], # 1-2
        [0, 0, 0, 1, 2, 3], # 1-0
        [0, 0, 0, 1, 2, 5], # 2-0
        [1, 0, 0, 1, 2, 6], # 2-3
        [0, 0, 1, 1, 2, 4], # 2-1
        [1, 1, 1, 1, 2, 7], # 3-4
        [0, 1, 0, 1, 2, 6], # 3-2
        [1, 0, 0, 1, 2, 7] # 4-3
        ], dtype=torch.float
    )

@pytest.fixture
def neighbour_attributes_no_edges():
    return torch.tensor([
        [0, 0, 1], # 0-1
        [0, 1, 0], # 0-2
        [0, 1, 0], # 1-2
        [0, 0, 0], # 1-0
        [0, 0, 0], # 2-0
        [1, 0, 0], # 2-3
        [0, 0, 1], # 2-1
        [1, 1, 1], # 3-4
        [0, 1, 0], # 3-2
        [1, 0, 0] # 4-3
        ], dtype=torch.float
    )

@pytest.fixture
def atom_batch_index():
    return torch.tensor([
        0, 0, 1, 1, 2, 2, 2, 3, 3, 4
    ], dtype=torch.long
    )

@pytest.fixture
def neighbour_indices():
    return torch.tensor([
        1, 2, 2, 0, 0, 3, 1, 4, 2, 3
    ], dtype=torch.long)

@pytest.fixture
def neighbour_counts():
    return torch.tensor([
        2, 2, 3, 2, 1
    ], dtype=torch.long)

@pytest.fixture
def case_data(
    node_attributes,
    edge_index,
    edge_attributes,
    neighbour_attributes,
    atom_batch_index,
    neighbour_indices,
    neighbour_counts
    ) -> CaseData:

    return CaseData(
        node_attributes,
        edge_index,
        edge_attributes,
        neighbour_attributes,
        atom_batch_index,
        neighbour_indices,
        neighbour_counts
    )

@pytest.fixture
def case_data_no_edge_attr(
    node_attributes,
    edge_index,
    edge_attributes,
    neighbour_attributes_no_edges,
    atom_batch_index,
    neighbour_indices,
    neighbour_counts
    ) -> CaseData:

    return CaseData(
        node_attributes,
        edge_index,
        edge_attributes,
        neighbour_attributes_no_edges,
        atom_batch_index,
        neighbour_indices,
        neighbour_counts
    )