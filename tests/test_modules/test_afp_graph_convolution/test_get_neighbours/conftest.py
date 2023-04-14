import pytest
from .case_data import CaseData
from ....modules import AFPGraphConvolution

import torch

l1 = AFPGraphConvolution(
    node_attribute_dim=3,
    edge_attribute_dim=3
)

x = torch.tensor(
    [
    [0]*3,
    [1]*3,
    [2]*3,
    [3]*3,
    [4]*3,
    [5]*3,
    [6]*3,
    [7]*3,
    [8]*3,
    ],
    dtype=torch.float32
    )

edge_index = torch.tensor(
    [
    [0, 2],
    [1, 2],
    [1, 0],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 8],
    [7, 6]
    ],
    dtype=torch.long
).T

edge_attributes = torch.tensor(
    [
    [1, 2, 3], 
    [1, 2, 4],
    [1, 2, 5],
    [1, 2, 6],
    [1, 2, 7],
    [2, 3, 1],
    [2, 4, 1],
    [2, 5, 1],
    ],
    dtype=torch.float32
)

node_index = torch.tensor(
    [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8],
    dtype=torch.long
)

neighbour_index = torch.tensor(
    [2, 1, 
     2, 0, 
     3, 0, 1, 
     4, 2, 
     3, 
     6, 
     8, 5, 7, 
     6, 
     6],
    dtype=torch.long
)

neighbour_counts = [2, 2, 3, 2, 1, 1, 3, 1, 1]

neighbour_x = x[neighbour_index]

neighbour_e = torch.tensor(
    [
    [1, 2, 3], [1, 2, 5], # 0-2, 0-1
    [1, 2, 4], [1, 2, 5], # 1-2, 1-0
    [1, 2, 6], [1, 2, 3], [1, 2, 4], # 2-3 2-0 2-1
    [1, 2, 7], [1, 2, 6], # 3-4 3-2
    [1, 2, 7], # 4-3
    [2, 3, 1], # 5-6
    [2, 4, 1], [2, 3, 1], [2, 5, 1], # 6-8 6-5 6-7
    [2, 5, 1], # 7-6
    [2, 4, 1], # 8-6
    ]
)

neighbour_xe = torch.concat([neighbour_x, neighbour_e], axis=1)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        x=x,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        neighbour_attributes=neighbour_xe,
        node_index=node_index,
        neighbour_index=neighbour_index,
        neighbour_counts=neighbour_counts
    ),
    CaseData(
        layer=l1,
        x=x,
        edge_index=edge_index,
        edge_attributes=None,
        neighbour_attributes=neighbour_x,
        node_index=node_index,
        neighbour_index=neighbour_index,
        neighbour_counts=neighbour_counts
    ),

]))
def case_data(request):
    return request.param