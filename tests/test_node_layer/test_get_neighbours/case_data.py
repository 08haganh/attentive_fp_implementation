
class CaseData(object):

    def __init__(
        self,
        x,
        edge_index,
        edge_attr,
        neighbour_node_attributes,
        neighbour_edge_attributes,
        neighbour_all_attributes,
        atom_batch_index,
        neighbour_counts
        ) -> None:

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.neighbour_node_attributes = neighbour_node_attributes
        self.neighbour_edge_attributes = neighbour_edge_attributes
        self.neighbour_all_attributes = neighbour_all_attributes
        self.atom_batch_index = atom_batch_index
        self.neighbour_counts = neighbour_counts
