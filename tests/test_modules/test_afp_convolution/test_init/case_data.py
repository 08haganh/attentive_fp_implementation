from .....modules import AFPConvolution

class CaseData():

    def __init__(
        self,
        layer: AFPConvolution,
        dim: int,
        p_dropout: float,
        ) -> None:

        self.layer = layer
        self.dim = dim
        self.p_dropout = p_dropout