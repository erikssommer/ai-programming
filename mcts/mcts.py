import random
import numpy as np
from node import Node


class MCTS:
    def __init__(self, root_node: Node, c_nn, dp_nn, epsilon, sigma, iterations):
        self.current_player = None
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

            # Change the current player
            self.change_current_player()

        return node.get_reward()

    def calculate_ucb1(self, node: Node):
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0:
            return np.inf
        elif self.current_player == 1:
            return self.get_max_value_move(node)
        else:
            return self.get_min_value_move(node)

    def get_max_value_move(self, node: Node):
        """
        Return the max value move for a given node and child
        """
        return node.rewards + self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def get_min_value_move(self, node: Node):
        """
        Return the min value move for a given node and child
        """
        return node.rewards - self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def select_best_child(self, node: Node):
        """
        Select the best child node using UCB1
        """
        best_score = -np.inf if self.current_player == 1 else np.inf
        best_child = None
        for child in node.children:
            if self.current_player == 1:
                ucb1 = self.calculate_ucb1(child)
                if ucb1 > best_score:
                    best_score = ucb1
                    best_child = child
            else:
                ucb1 = self.calculate_ucb1(child)
                if ucb1 < best_score:
                    best_score = ucb1
                    best_child = child
        return best_child

    def node_expansion(self, node: Node):
        # Expand node by adding one of its unexpanded children
        # Get the legal moves from the current state
        legal_moves = node.get_legal_moves()

        # Expand the node by creating child nodes for each legal move
        for move in legal_moves:
            node.apply_action(move)

        return (node.children[0] if len(node.children) > 0 else None)

    def simulate(self, node: Node):
        if random.random() < self.sigma:
            return self.default_policy_rollout(node)
        else:
            return self.chritic(node)

    def chritic(self, node: Node):
        # TODO: Use the chritic neural network to simulate a playout from the current node
        pass

    def backpropagate(self, node: Node, reward):
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def tree_search(self, node: Node):
        # Test if node is terminal
        if node.is_game_over():
            return node

        while len(node.children) != 0:
            node = self.select_best_child(node)

            # Change the current player
            self.change_current_player()

        if node.visits != 0:
            child = self.node_expansion(node)

        return child

    def change_current_player(self):
        self.current_player = self.current_player % 2 + 1

    def get_best_move(self) -> Node:
        return max(self.root.children, key=lambda c: c.visits)

    def get_distribution(self):
        total_visits = sum(child.visits for child in self.root.children)
        return [(child.state, child.visits / total_visits) for child in self.root.children]

    def search(self, current_node, starting_player=1) -> Node:
        # Reset the root node
        if current_node is not None:
            self.root = current_node

        for _ in range(self.iterations):
            node: Node = self.root
            self.current_player = starting_player
            leaf_node = self.tree_search(node, starting_player)
            reward = self.simulate(leaf_node, starting_player)
            self.backpropagate(leaf_node, reward)
        # Use the edge (from the root) with the highest visit count as the actual move.
        best_move = self.get_best_move()
        distribution = self.get_distribution()
        return best_move, distribution
