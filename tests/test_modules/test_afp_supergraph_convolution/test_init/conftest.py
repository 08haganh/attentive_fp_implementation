import pytest
from .case_data import CaseData
from .....modules import AFPSuperGraphConvolution

l1 = AFPSuperGraphConvolution(
    dim=10,
    p_dropout=0.24
)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        dim=10,
        p_dropout=0.24
    )
]))
def case_data(request):
    return request.param

