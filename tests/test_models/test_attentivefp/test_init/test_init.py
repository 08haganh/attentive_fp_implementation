from .case_data import CaseData

def test_init(case_data: CaseData):

    # init args
    _test_init_args(case_data)
    _test_layer_sizes(case_data)
    _test_graph_layers(case_data)
    _test_supergraph_layers(case_data)
    _test_linear_layers(case_data)
    _test_dropout(case_data)
    _test_graph_descriptors_embedding_layers(case_data)
    

def _test_init_args(case_data: CaseData):
    assert case_data.model.node_attribute_dim == case_data.node_attribute_dim
    assert case_data.model.edge_attribute_dim == case_data.edge_attribute_dim
    assert case_data.model.embedding_dim == case_data.embedding_dim
    assert case_data.model.p_dropout == case_data.p_dropout
    assert case_data.model.output_dim == case_data.output_dim
    assert case_data.model.n_graph_layers == case_data.n_graph_layers
    assert case_data.model.n_supergraph_layers == case_data.n_supergraph_layers
    assert case_data.model.n_linear_layers == case_data.n_linear_layers
    assert case_data.model.n_graph_descriptors == case_data.n_graph_descriptors
    assert case_data.model.graph_descriptors_dims == case_data.graph_descriptors_dims

def _test_layer_sizes(case_data: CaseData):
    assert len(case_data.model.graph_layers) == case_data.n_graph_layers
    assert len(case_data.model.supergraph_layers) == case_data.n_supergraph_layers
    assert len(case_data.model.linear_layers) == case_data.n_linear_layers

def _test_graph_layers(case_data: CaseData):
    l0 = case_data.model.graph_layers[0]
    # init args
    assert l0.node_attribute_dim == case_data.node_attribute_dim
    assert l0.edge_attribute_dim == case_data.edge_attribute_dim
    assert l0.embedding_dim == case_data.embedding_dim
    assert l0.p_dropout == case_data.p_dropout

    if len(case_data.model.graph_layers) > 1:
        for l in case_data.model.graph_layers[1:]:
            assert l.node_attribute_dim == case_data.embedding_dim
            assert l.edge_attribute_dim == None
            assert l.embedding_dim == None
            assert l.p_dropout == case_data.p_dropout

def _test_supergraph_layers(case_data: CaseData):
    for l in case_data.model.supergraph_layers:
        assert l.dim == case_data.embedding_dim
        assert l.p_dropout == case_data.p_dropout

def _test_linear_layers(case_data: CaseData):
    for l in case_data.model.linear_layers[:-1]:
        assert l.in_features == case_data.embedding_dim
        assert l.out_features == case_data.embedding_dim
    
    lm1 = case_data.model.linear_layers[-1]
    assert lm1.in_features == case_data.embedding_dim
    assert lm1.out_features == case_data.output_dim

def _test_dropout(case_data: CaseData):
    assert case_data.model.dropout.p == case_data.p_dropout

def _test_graph_descriptors_embedding_layers(case_data: CaseData):
    if case_data.n_graph_descriptors is None:
        assert case_data.model.graph_descriptor_embedding_layers is None
    else:
        for i, l in enumerate(case_data.model.graph_descriptor_embedding_layers):
            assert l.in_features == case_data.graph_descriptors_dims[i]
            assert l.out_features == case_data.embedding_dim

