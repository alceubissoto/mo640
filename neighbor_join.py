from skbio import DistanceMatrix
from skbio.tree import nj


class NeighborJoinRunner:
    genotype = []
    next_node_index = 0
    dist_matrix = None

    def __init__(self, data, input_amount):
        self.next_node_index = input_amount
        self.__load_distance_matrix(data)

    def __post_order(self, tree_node):
        for child in tree_node.children:
            self.__post_order(child)

        if tree_node.name is None:
            tree_node.name = str(self.next_node_index)
            self.next_node_index += 1

    def __build_genotype(self, tree_node):
        if tree_node.is_tip():
            return

        for child in tree_node.children:
            self.__build_genotype(child)

        # build genotype
        count_visited = 0
        for child in tree_node.children:
            self.genotype.append(int(child.name))

            # the cgp structure allows only binary nodes
            count_visited += 1
            if count_visited % 2 == 0:
                self.genotype.append(0)

        # one node lingering in a tuple
        if count_visited % 2 == 1:
            self.genotype.append(self.next_node_index)
            self.genotype.append(0)

    def get_individual(self):
        individual = dict()
        individual['genotype'] = self.genotype
        individual['output'] = int(self.next_node_index - 1)

        return individual

    def get_dist_matrix(self):
        return self.dist_matrix

    def __load_distance_matrix(self, data):
        dm = DistanceMatrix(data)
        nj_tree = nj(dm)

        df = nj_tree.tip_tip_distances().to_data_frame()
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        self.dist_matrix = df

        nj_tree.bifurcate()
        self.__post_order(nj_tree)
        self.__build_genotype(nj_tree)

if __name__ == "__main__":
    data = [[0, 12, 12, 12, 12],
           [12, 0, 4, 6, 6],
           [ 12, 4, 0, 6, 6],
           [12, 6, 6, 0, 2],
           [ 12, 6, 6, 2, 0]]
    dm = DistanceMatrix(data)
    tree = nj(dm)
    tree.bifurcate()
    print(tree.ascii_art())
    runner = NeighborJoinRunner()
    runner.__post_order(tree)
    runner.__build_genotype(tree)
    print(runner.genotype)