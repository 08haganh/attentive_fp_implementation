

class CaseData(object):

    def __init__(
        self,
        node_layer,
        node_attribute_dim,
        edge_attribute_dim,
        embedding_dim,
        p_dropout,
        embed
    ) -> None:
        
        self.node_layer = node_layer
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout
        self.embed = embed