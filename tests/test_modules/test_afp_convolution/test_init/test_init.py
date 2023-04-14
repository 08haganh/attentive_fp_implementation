from .case_data import CaseData

def test_init(case_data: CaseData):

    # init args
    assert case_data.layer.dim == case_data.dim
    assert case_data.layer.p_dropout == case_data.p_dropout

    # layers
    assert case_data.layer.alignment_layer.in_features == 2 * case_data.dim
    assert case_data.layer.alignment_layer.out_features == 1
    assert case_data.layer.context_layer.in_features == case_data.dim
    assert case_data.layer.context_layer.out_features == case_data.dim
    assert case_data.layer.readout_layer.input_size == case_data.dim
    assert case_data.layer.readout_layer.hidden_size == case_data.dim
    assert case_data.layer.dropout.p == case_data.p_dropout

    # attentions
    assert case_data.layer.attentions is None
    assert case_data.layer.node_index is None
    assert case_data.layer.neighbour_index is None