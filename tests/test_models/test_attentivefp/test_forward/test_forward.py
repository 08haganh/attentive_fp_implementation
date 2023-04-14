from .case_data import CaseData

def test_forward(case_data: CaseData):
    
    preds = case_data.model(
        case_data.x,
        case_data.edge_index,
        case_data.edge_attr,
        case_data.batch_index,
        case_data.graph_x
    )

    assert preds.shape == case_data.output_dim