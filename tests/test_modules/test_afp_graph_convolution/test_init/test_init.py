from .case_data import CaseData

def test_init(case_data: CaseData):

    # init args
    dim = case_data.node_attribute_dim if case_data.embedding_dim is None else case_data.embedding_dim
    assert case_data.layer.node_attribute_dim == case_data.node_attribute_dim
    assert case_data.layer.edge_attribute_dim == case_data.edge_attribute_dim
    assert case_data.layer.embedding_dim == case_data.embedding_dim
    assert case_data.layer.p_dropout == case_data.p_dropout
    assert case_data.layer.dim == dim

    # layers
    if case_data.edge_attribute_dim is not None:
        neighbour_feature_dim = case_data.node_attribute_dim + case_data.edge_attribute_dim
    else:
        neighbour_feature_dim = case_data.node_attribute_dim

    if case_data.embedding_dim is not None:
        assert case_data.layer.node_embedding_layer.in_features == case_data.node_attribute_dim
        assert case_data.layer.node_embedding_layer.out_features == case_data.embedding_dim
        assert case_data.layer.neighbour_embedding_layer.in_features == neighbour_feature_dim
        assert case_data.layer.neighbour_embedding_layer.out_features == case_data.embedding_dim

    # attentions
    assert case_data.layer.attentions is None
    assert case_data.layer.node_index is None
    assert case_data.layer.neighbour_index is None
        
