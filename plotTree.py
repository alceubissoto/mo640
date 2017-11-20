import pygraphviz as pgv

# Get dict of edges, weights and plot a tree
def plotTree(edges, num_inputs):
    G = pgv.AGraph(splines=False)

    for edge in edges.keys():
        G.add_node(edge[0], shape='circle' if edge[0] < num_inputs else 'point')
        G.add_node(edge[1], shape='circle' if edge[1] < num_inputs else 'point')

    for edge, weight in edges.items():
        G.add_edge(edge[0], edge[1], label=str(weight), color='gray')

    G.draw('tree.png', prog='dot') # draw png
