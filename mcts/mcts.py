import random
import numpy as np
from node import Node


class MCTS:
    def __init__(self, root_node: Node, c_nn, dp_nn, epsilon=1.0, sigma=2.0, iterations=1000):
        self.iterations = iterations
        self.root = root_node
        self.dp_nn = dp_nn
        self.epsilon = epsilon
        self.sigma = sigma
        self.c_nn = c_nn
        self.c = 1.4

    def default_policy_rollout(self, node: Node):
        """
        Rollout function using epsilon-greedy strategy with default policy

        Parameters:
        node (object): the current node state of the game
        default_policy (network): the default policy network

        Returns:
        float: the cumulative reward obtained in the rollout
        """
        while node.is_game_over():
            legal_moves = node.get_legal_moves()

            if not legal_moves:
                break

            if random.random() < self.epsilon:
                next_move = random.choice(legal_moves)
            else:
                # TODO: get the best move from the default policy network
                pass
            parent_node = node
            # Apply the action to the node and get back the next node
            node = node.apply_action(next_move)
            node.parent = parent_node

        return node.get_reward()

    def calculate_ucb1(self, node: Node, player):
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0:
            return np.inf
        elif player == 1:
            return self.get_max_value_move(node, node)
        else:
            return self.get_min_value_move(node, node)

    def get_max_value_move(self, node: Node):
        return node.rewards + self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def get_min_value_move(self, node: Node):
        return node.rewards - self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def select_best_child(self, node: Node, player):
        # Select child with highest UCB1 value
        best_score = -np.inf
        best_child = None
        for child in node.children:
            ucb1 = self.calculate_ucb1(child, player)
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

    def simulate(self, node: Node, player):
        if random.random() < self.sigma:
            return self.default_policy_rollout(node, player)
        else:
            return self.chritic(node, player)

    def chritic(self, node: Node, player):
        # TODO: Use the chritic neural network to simulate a playout from the current node
        pass

    def backpropagate(self, node: Node, reward):
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def tree_search(self, node: Node, player):
        while len(node.children) != 0:
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
        # Use the edge (from the root) with the highest visit count as the actual move.
        return max(self.root.children, key=lambda c: c.visits)
