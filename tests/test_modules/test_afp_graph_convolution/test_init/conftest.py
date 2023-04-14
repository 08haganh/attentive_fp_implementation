import pytest

from .case_data import CaseData
from .....modules import AFPGraphConvolution

# default init
l1 = AFPGraphConvolution(
    node_attribute_dim=10
)

# embedding_dim no edge_attr
l2 = AFPGraphConvolution(
    node_attribute_dim=15,
    embedding_dim=20
)

## embedding_dim and edge_attr
l3 = AFPGraphConvolution(
    node_attribute_dim=20,
    edge_attribute_dim=5,
    embedding_dim=64
)

# using all init args
l4 = AFPGraphConvolution(
    node_attribute_dim=20,
    edge_attribute_dim=5,
    embedding_dim=64,
    p_dropout=0.8
)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        node_attribute_dim=10,
        edge_attribute_dim=None,
        embedding_dim=None,
        p_dropout=0
    ),
    CaseData(
        layer=l2,
        node_attribute_dim=15,
        edge_attribute_dim=None,
        embedding_dim=20,
        p_dropout=0
    ),
    CaseData(
        layer=l3,
        node_attribute_dim=20,
        edge_attribute_dim=5,
        embedding_dim=64,
        p_dropout=0
    ),
    CaseData(
        layer=l4,
        node_attribute_dim=20,
        edge_attribute_dim=5,
        embedding_dim=64,
        p_dropout=0.8
    ),
]))
def case_data(request):
    return request.param