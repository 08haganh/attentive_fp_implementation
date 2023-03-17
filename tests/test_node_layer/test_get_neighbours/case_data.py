
class CaseData(object):

    def __init__(
        self,
        graph,
        neighbour_attributes,
        atom_batch_index,
        neighbour_counts
        ) -> None:

        self.graph = graph
        self.neighbour_attributes = neighbour_attributes
        self.atom_batch_index = atom_batch_index
        self.neighbour_counts = neighbour_counts
