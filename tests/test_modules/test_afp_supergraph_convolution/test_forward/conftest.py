import pytest
import torch

from .case_data import CaseData
from .....modules import AFPSuperGraphConvolution

torch.manual_seed(0)

l1 = AFPSuperGraphConvolution(
    dim=3
)

'''
For testing the forward pass, we are going to to use to cases, one using all
defaults, and other where all args are passed
'''

x = torch.tensor([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5]
    ],
    dtype=torch.float32
)

batch_index = torch.tensor([
    0, 0, 0, 1, 1
], dtype=torch.long)

graph_embeddings = torch.tensor([
    [10, 10, 10],
    [12, 12, 12]
], dtype=torch.float32)

graph_x = torch.tensor(
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

readout_shape0 = torch.Size([1, 3])
expected_attentions0 = torch.tensor([
    0.2, 0.2, 0.2, 0.2, 0.2
], dtype=torch.float32).reshape(-1, 1)
node_index0 = torch.tensor([
    0, 0, 0, 0, 0
], dtype=torch.long)
neighbour_index0 =  torch.tensor([
    0, 1, 2, 3, 4
], dtype=torch.long)

readout_shape1 = torch.Size([2, 3])
expected_attentions1 = torch.tensor([
    1/5, 1/5, 1/5, 1/5, 1/5, 1/4, 1/4, 1/4, 1/4
], dtype=torch.float32).reshape(-1, 1)
node_index1 = torch.tensor([
    0, 0, 0, 0, 0, 1, 1, 1, 1
], dtype=torch.long)
neighbour_index1 =  torch.tensor([
    0, 1, 2, 3, 4, 5, 6, 7, 8
], dtype=torch.long)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        x=x,
        batch_index=None,
        graph_embeddings=None,
        graph_x=None,
        readout_shape=readout_shape0,
        expected_attentions=expected_attentions0,
        node_index=node_index0,
        neighbour_index=neighbour_index0
    ),
    CaseData(
        layer=l1,
        x=x,
        batch_index=batch_index,
        graph_embeddings=graph_embeddings,
        graph_x=graph_x,
        readout_shape=readout_shape1,
        expected_attentions=expected_attentions1,
        node_index=node_index1,
        neighbour_index=neighbour_index1
    ),
]))
def case_data(request):
    return request.param


