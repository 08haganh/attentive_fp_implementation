from ....modules import AFPSuperGraphConvolution

class CaseData:

    def __init__(
        self,
        layer: AFPSuperGraphConvolution,
        dim: int,
        p_dropout: float
        ) -> None:

        self.layer = layer
        self.dim = dim
        self.p_dropout = p_dropout