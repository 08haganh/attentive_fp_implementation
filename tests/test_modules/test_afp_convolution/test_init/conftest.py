import pytest

from .case_data import CaseData
from ....modules import AFPConvolution

# defaults
l1 = AFPConvolution(
    dim=10,
)

# with p_dropout
l2 = AFPConvolution(
    dim=10,
    p_dropout=0.5
)

@pytest.fixture(params=([
    CaseData(
        layer=l1,
        dim=10,
        p_dropout=0
    ),
    CaseData(
        layer=l2,
        dim=10,
        p_dropout=0.5
    )
]))
def case_data(request):
    return request.param