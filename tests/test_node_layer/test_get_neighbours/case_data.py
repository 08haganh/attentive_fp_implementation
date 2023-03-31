
class CaseData(object):

    def __init__(
        self,
        x,
        edge_index,
        edge_attr,
        neighbour_attributes,
        atom_batch_index,
        neighbour_indices,
        neighbour_counts
        ) -> None:

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.neighbour_attributes = neighbour_attributes
        self.atom_batch_index = atom_batch_index
        self.neighbour_indices = neighbour_indices
        self.neighbour_counts = neighbour_counts
