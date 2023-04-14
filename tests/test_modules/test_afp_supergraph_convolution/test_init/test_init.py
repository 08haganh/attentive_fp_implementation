from .case_data import CaseData

def test_init(case_data: CaseData):

    assert case_data.layer.dim == case_data.dim
    assert case_data.layer.p_dropout == case_data.p_dropout