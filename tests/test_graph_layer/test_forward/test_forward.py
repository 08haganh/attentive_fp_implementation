from ....modules.graph_layer import GraphLayer
import torch

def test_forward(case_data):

    graph_layer = GraphLayer(dim=3, p_dropout=0.2)

    _ = graph_layer(case_data.x, case_data.batch_index)

    assert torch.all(graph_layer.graph_nodes == case_data.graph_nodes)
    assert torch.all(graph_layer.joint_attributes == case_data.joint_attributes)