import random
import numpy as np
from node import Node


class MCTS:
    def __init__(self, state, nnet, iterations=1000):
        self.iterations = iterations
        self.root = Node(state)
        self.nnet = nnet
        self.c = 1.4

    def random_playout(self, node: Node):
        """
        Default rollout policy: play randomly until the end of the game
        """
        current_state = node.state
        while not current_state.is_terminal():
            state = random.choice(current_state.get_children())
        return state.get_reward()

    def calculate_ucb1(self, node: Node, child: Node, player):
        """
        Calculate UCB1 value for a given node and child
        """
        if child.visits == 0:
            return np.inf
        elif player == 1:
            return self.get_max_value_move(node, child)
        else:
            return self.get_min_value_move(node, child)

    def get_max_value_move(self, node: Node, child: Node):
        return child.rewards + self.c * np.sqrt(np.log(node.visits) / (1 + child.visits))

    def get_min_value_move(self, node: Node, child: Node):
        return child.rewards - self.c * np.sqrt(np.log(node.visits) / (1 + child.visits))

    def select_best_child(self, node: Node, player):
        # Select child with highest UCB1 value
        best_score = -np.inf
        best_child = None
        for child in node.children:
            ucb1 = self.calculate_ucb1(node, child, player)
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        return best_child

    def node_expansion(self, node: Node):
        # Expand node by adding one of its unexpanded children
        unexpanded_children = [
            child for child in node.state.get_children()
            if child not in [c.state for c in node.children]
        ]

        if not unexpanded_children:
            return None

        selected_child = random.choice(unexpanded_children)
        return node.add_child(selected_child)

    def simulate(self, node: Node, player, epsilon=1.0):
        if random.random() < epsilon:
            return self.random_playout(node)
        else:
            return self.chritic(node, player)

    def chritic(self, node: Node, player):
        # Use the neural network to simulate a playout from the current node
        state = node.get_state()
        split_state = np.concatenate(
            ([player], [int(i) for i in state.split()]))
        preds = self.nnet.predict(np.array([split_state]))
        return self.nnet.best_action(preds[0])

    def backpropagate(self, node: Node, reward):
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def tree_search(self, node: Node, player):
        while node.children != 0:
            node = self.select_best_child(node, player)

        if node.visits != 0:
            child = self.node_expansion(node)

        return child

    def search(self, player=1) -> Node:
        for _ in range(self.iterations):
            node: Node = self.root
            leaf_node = self.tree_search(node, player)
            reward = self.simulate(leaf_node, player)
            self.backpropagate(leaf_node, reward)
        return max(self.root.children, key=lambda c: c.visits)
