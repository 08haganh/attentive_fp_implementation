def test_init(case_data):

    assert case_data.node_layer.node_attribute_dim == case_data.node_attribute_dim
    assert case_data.node_layer.edge_attribute_dim == case_data.edge_attribute_dim
    assert case_data.node_layer.embedding_dim == case_data.embedding_dim
    assert case_data.node_layer.p_dropout == case_data.p_dropout
    assert case_data.node_layer.embed == case_data.embed