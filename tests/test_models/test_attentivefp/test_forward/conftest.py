import pytest
from .case_data import CaseData
from .....models import AttentiveFP
import torch

x = torch.tensor([
    [0]*3,
    [1]*3,
    [2]*3,
    [3]*3,
    [4]*3,
    [5]*3,
    [6]*3
], dtype=torch.float32)

edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 6],
    [1, 2, 0, 4, 5, 4]
], dtype=torch.long)

edge_attr= torch.tensor([
    [9],
    [8],
    [7],
    [6],
    [5],
    [4],
], dtype=torch.float32)

batch_index = torch.tensor([
    0, 0, 0, 1, 1, 1, 1
], dtype=torch.long)

graph_x = [
    [torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32), 
     torch.tensor([6, 5, 4], dtype=torch.float32)],
    [torch.tensor([9, 8, 7, 6, 5], dtype=torch.float32), 
     torch.tensor([4, 5, 6], dtype=torch.float32)]
]

m1 = AttentiveFP(
    node_attribute_dim=3,
    embedding_dim=32,
    output_dim=1
)

m2 = AttentiveFP(
    node_attribute_dim=3,
    edge_attribute_dim=1,
    embedding_dim=32,
    output_dim=4,
    n_graph_layers=3,
    n_supergraph_layers=3,
    n_linear_layers=3,
    n_graph_descriptors=2,
    graph_descriptors_dims=[5, 3]
)

@pytest.fixture(params=([
    CaseData(
        model=m1,
        x=x,
        edge_index=edge_index,
        output_dim=torch.Size([1, 1])
    ),
    CaseData(
        model=m2,
        x=x,
        edge_index=edge_index,
        output_dim=torch.Size([2, 4]),
        edge_attr=edge_attr,
        batch_index=batch_index,
        graph_x=graph_x
    ),
]))
def case_data(request):
    return request.param