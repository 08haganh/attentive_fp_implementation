import pytest
from.case_data import CaseData
from .....models import AttentiveFP

# defaults
m1 = AttentiveFP(
    node_attribute_dim=10
)

m2 = AttentiveFP(
    node_attribute_dim=20,
    edge_attribute_dim=5,
    embedding_dim=32,
    p_dropout=0.3,
    output_dim=2,
    n_graph_descriptors=3,
    n_supergraph_layers=3,
    n_linear_layers=5,
    n_graph_layers=3,
    graph_descriptors_dims=[20, 32, 128]
)

@pytest.fixture(params=([
    CaseData(
        model=m1,
        node_attribute_dim=10,
        edge_attribute_dim=None,
        embedding_dim=256,
        p_dropout=0,
        output_dim=1,
        n_graph_layers=1,
        n_supergraph_layers=1,
        n_linear_layers=1,
        n_graph_descriptors=None,
        graph_descriptors_dims=None
    ),
    CaseData(
        model=m2,
        node_attribute_dim=20,
        edge_attribute_dim=5,
        embedding_dim=32,
        p_dropout=0.3,
        output_dim=2,
        n_graph_layers=3,
        n_supergraph_layers=3,
        n_linear_layers=5,
        n_graph_descriptors=3,
        graph_descriptors_dims=[20, 32, 128]
    )
]))
def case_data(request):
    return request.param