

class CaseData(object):

    def __init__(
        self,
        x,
        batch_index,
        graph_nodes,
        joint_attributes
        ) -> None:

        self.x = x
        self.batch_index = batch_index
        self.graph_nodes = graph_nodes
        self.joint_attributes = joint_attributes
