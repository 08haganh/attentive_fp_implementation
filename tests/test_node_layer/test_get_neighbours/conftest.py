import pytest
from .case_data import CaseData

import numpy as np

@pytest.fixture
def node_attributes():
    return np.array([
        [0, 0 ,0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0 ,0],
        [1, 1, 1]
    ])

@pytest.fixture
def edge_index():
    return np.array([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 3, 4]
    ])

@pytest.fixture
def edge_attributes():
    return np.array([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 2, 6],
        [1, 2, 7]
    ])

@pytest.fixture
def neighbour_node_attributes():
    return np.array([
        [0, 0, 1], # 0-1
        [0, 1, 0], # 0-2
        [0, 0 ,0], # 1-0
        [0, 1, 0], # 1-2
        [0, 0 ,0], # 2-0
        [0, 0 ,1], # 2-1
        [1, 0, 0], # 2-3
        [0, 1, 0], # 3-2
        [1, 1, 1], # 3-4
        [1, 0, 0]  # 4-3
        ]
    )

@pytest.fixture
def neighbour_edge_attributes():
    return np.array([
        [1, 2, 3], # 0-1
        [1, 2, 5], # 0-2
        [1, 2, 3], # 1-0
        [1, 2, 4], # 1-2
        [1, 2, 3], # 2-0
        [1, 2, 4], # 2-1
        [1, 2, 6], # 2-3
        [1, 2, 6], # 3-2
        [1, 2, 7], # 3-4
        [1, 2, 7]  # 4-3
    ])

@pytest.fixture
def neighbour_all_attributes(
    neighbour_node_attributes, 
    neighbour_edge_attributes
    ):
    return np.concatenate(
        [neighbour_node_attributes,
        neighbour_edge_attributes
        ],
        axis=1)

@pytest.fixture
def atom_batch_index():
    return np.array([
        0, 0, 1, 1, 2, 2, 2, 3, 3, 4
    ])

@pytest.fixture
def neighbour_counts():
    return [
        2, 2, 3, 2, 1
    ]

