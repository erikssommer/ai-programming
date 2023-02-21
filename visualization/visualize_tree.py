import graphviz

# For visualization purposes


class VisualizeTree:

    # Create init method
    def __init__(self, node):
        self.dot = graphviz.Digraph()
        self.node = node

    def _visualize_tree(self, dot, parent, node):
        dot.node(
            str(node), label=f"{str(node)}\nvisits: {node.visits}\nrewards: {node.rewards}")

        if parent is not None:
            dot.edge(str(parent), str(node))

        for child in node.children:
            self._visualize_tree(dot, node, child)

    def visualize_tree(self):
        self._visualize_tree(self.dot, None, self.node)
        self.dot.render('tree.gv', view=True)
