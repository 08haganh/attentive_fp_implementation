
class CaseData(object):

    def __init__(
        self,
        x,
        edge_index,
        edge_attr,
        joint_attributes
    ) -> None:
        
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.joint_attributes = joint_attributes