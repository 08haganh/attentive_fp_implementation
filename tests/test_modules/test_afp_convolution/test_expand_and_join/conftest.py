import pytest
from .case_data import CaseData
from .....modules import AFPConvolution

import torch

node_embeddings = torch.tensor([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    ],
    dtype=torch.float32
)

neighbour_attributes = torch.tensor([
    [2, 2, 3], [2, 3, 2], [3, 2, 2],
    [3, 3, 2],
    [4, 4, 2],[4, 2, 4]
    ],
    dtype=torch.float32
)

neighbour_counts = [3, 1, 2]

l1 = AFPConvolution(
    dim=10
)

joint_attributes = torch.tensor([
    [0, 0, 1, 2, 2, 3],
    [0, 0, 1, 2, 3, 2],
    [0, 0, 1, 3, 2, 2],
    [0, 1, 0, 3, 3, 2],
    [1, 0, 0, 4, 4, 2],
    [1, 0, 0, 4, 2, 4],
])

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        node_embeddings=node_embeddings,
        neighbour_attributes=neighbour_attributes,
        neighbour_counts=neighbour_counts,
        joint_attributes=joint_attributes
    )
]))
def case_data(request):
    return request.param