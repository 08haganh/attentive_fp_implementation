import pytest
from .case_data import CaseData
from ....modules.node_layer import NodeLayer

@pytest.fixture(
        params=(
            [CaseData(
                NodeLayer(100, 50, 200, 0.3),
                100, 50, 200, 0.3, True
                ),
            CaseData(
                NodeLayer(64, 50, None, 0.5),
                64, 50, None, 0.5, False
                )
            ]
            )
        )
def case_data(request):
    return request.param

