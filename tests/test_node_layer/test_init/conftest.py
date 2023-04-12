import pytest
from .case_data import CaseData
from ....modules.node_layer import NodeLayer

@pytest.fixture(
        params=(
            [CaseData(
                node_layer=NodeLayer(
                    node_attribute_dim=100, 
                    edge_attribute_dim=50, 
                    embedding_dim=200, 
                    p_dropout=0.3),
                node_attribute_dim=100, 
                edge_attribute_dim=50, 
                embedding_dim=200, 
                p_dropout=0.3, 
                embed=True, 
                dim=200
                ),
            CaseData(
                node_layer=NodeLayer(
                    node_attribute_dim=64, 
                    edge_attribute_dim=50, 
                    embedding_dim=None, 
                    p_dropout=0.5),
                node_attribute_dim=64, 
                edge_attribute_dim=50, 
                embedding_dim=None, 
                p_dropout=0.5, 
                embed=False, 
                dim=64
                )
            ]
            )
        )
def case_data(request):
    return request.param

