import warnings

from .case_data import CaseData
import torch
def test_forward(case_data: CaseData):
    
    preds = case_data.model(
        case_data.x,
        case_data.edge_index,
        case_data.edge_attr,
        case_data.batch_index,
        case_data.graph_x
    )

    assert preds.shape == case_data.output_dim

def test_forward_gpu(case_data: CaseData):
    if torch.cuda.is_available():
        model = case_data.to('cuda')
        x = case_data.x.to('cuda')
        edge_index = case_data.edge_index.to('cuda')
        edge_attr = case_data.edge_attr.to('cuda')
        batch_index = case_data.batch_index.to('cuda')
        graph_x = [
            [x[0].to('cuda'), x[1].to('cuda')]
            for x in case_data.graph_x
        ]

        preds = model(
            x,
            edge_index,
            edge_attr,
            batch_index,
            graph_x
            )

        assert preds.shape == case_data.output_dim
    else:
        warnings.warn(UserWarning('gpus not available, forward pass not tested for gpus'))