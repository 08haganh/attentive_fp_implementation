import pytest
from .case_data import CaseData
from .....modules import AFPGraphConvolution

import torch

torch.manual_seed(0)

# if embedding_dim is None,
# edge_attr must be None on forward pass
l1 = AFPGraphConvolution(
    node_attribute_dim=3,
)

l2 = AFPGraphConvolution(
    node_attribute_dim=3,
    edge_attribute_dim=3,
    embedding_dim=32
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

expected_attentions = torch.concat([
    torch.tensor([1/n]*n, dtype=torch.float32) for n in neighbour_counts
    ]
).reshape(-1, 1)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        x=x,
        edge_index=edge_index,
        edge_attr=None,
        readout_shape=torch.Size([9, 3]),
        expected_attentions=expected_attentions,
        node_index=node_index,
        neighbour_index=neighbour_index
    ),
    CaseData(
        layer=l2,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attributes,
        readout_shape=torch.Size([9, 32]),
        expected_attentions=expected_attentions,
        node_index=node_index,
        neighbour_index=neighbour_index
    )
]))
def case_data(request):
    return request.param