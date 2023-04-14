import pytest
import torch

from .case_data import CaseData
from ....modules import AFPSuperGraphConvolution

l1 = AFPSuperGraphConvolution(
    dim=3
)

x = torch.tensor([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5]
    ],
    dtype=torch.float32
)

batch_index1 = torch.tensor([
    0, 0, 0, 0, 0
], dtype=torch.long)

batch_index2 = torch.tensor([
    0, 0, 0 , 1, 1 
], dtype=torch.long)

graph_x1 = torch.tensor(
    [
        [
            [1, 2, 3], [3, 2, 1]
        ]
    ], 
    dtype=torch.float32
)

graph_x2 = torch.tensor(
    [
        [
            [1, 2, 3], [3, 2, 1]
        ],
        [
            [6, 5, 4], [4, 5, 6]
        ]
    ], 
    dtype=torch.float32
)

neighbour_attributes0 = x
return_batch_index0 = batch_index1
neighbour_index0 = torch.tensor([
    0, 1, 2, 3, 4
], dtype=torch.long)
neighbour_counts0 = [5]

neighbour_attributes1 = torch.tensor([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5],
    [1, 2, 3], 
    [3, 2, 1]
    ],
    dtype =torch.float32
)

neighbour_attributes2 = torch.tensor([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [1, 2, 3], 
    [3, 2, 1],
    [4, 4, 4],
    [5, 5, 5],
    [6, 5, 4], 
    [4, 5, 6]
    ],
    dtype =torch.float32
)

return_batch_index1 = torch.tensor([
    0, 0, 0, 0, 0, 0, 0
], dtype=torch.long)

return_batch_index2 = torch.tensor([
    0, 0, 0, 0, 0, 1, 1, 1, 1
], dtype=torch.long)

neighbour_index1 = torch.tensor([
    0, 1, 2, 3, 4, 5, 6
], dtype=torch.long)

neighbour_index2 = torch.tensor([
    0, 1, 2, 3, 4, 5, 6, 7, 8
], dtype=torch.long)

neighbour_counts1 = [7]

neighbour_counts2 = [5, 4]

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        x=x,
        batch_index=batch_index1,
        graph_x=None,
        neighbour_attributes=neighbour_attributes0,
        return_batch_index=return_batch_index0,
        neighbour_index=neighbour_index0,
        neighbour_counts=neighbour_counts0
    ),
    CaseData(
        layer=l1,
        x=x,
        batch_index=batch_index1,
        graph_x=graph_x1,
        neighbour_attributes=neighbour_attributes1,
        return_batch_index=return_batch_index1,
        neighbour_index=neighbour_index1,
        neighbour_counts=neighbour_counts1
    ),
    CaseData(
        layer=l1,
        x=x,
        batch_index=batch_index2,
        graph_x=graph_x2,
        neighbour_attributes=neighbour_attributes2,
        return_batch_index=return_batch_index2,
        neighbour_index=neighbour_index2,
        neighbour_counts=neighbour_counts2
    ),
]))
def case_data(request):
    return request.param