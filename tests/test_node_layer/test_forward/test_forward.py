from ....modules.node_layer import NodeLayer

import torch

def test_forward(case_data):
    
    node_layer = NodeLayer(
        node_attribute_dim=3,
        edge_attribute_dim=3,
        embedding_dim=None,
        p_dropout=0.2
    )

    _ = node_layer(case_data.x, case_data.edge_index, case_data.edge_attr)
    print(node_layer.joint_attributes)
    print(case_data.joint_attributes)
    assert torch.all(node_layer.joint_attributes == case_data.joint_attributes)