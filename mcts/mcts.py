import copy
import random
from typing import Tuple, List, Any, Union

import numpy as np
from mcts.node import Node

import torch


class MCTS:
    def __init__(self, root_node: Node, epsilon, sigma, iterations, c, c_nn=None, dp_nn=None):
        self.iterations = iterations
        self.root = root_node
        self.dp_nn = dp_nn
        self.epsilon = epsilon
        self.sigma = sigma
        self.c_nn = c_nn
        self.c = c

    def default_policy_rollout(self, node: Node) -> int:
        """
        Rollout function using epsilon-greedy strategy with default policy

        Parameters:
        node (object): the current node state of the game
        default_policy (network): the default policy network

        Returns:
        float: the cumulative reward obtained in the rollout
        """

        while not node.is_game_over():
            # Get the legal moves for the current state
            legal_moves = node.get_legal_moves()

            if random.random() < self.epsilon:
                next_move = random.choice(legal_moves)
            else:
                state = torch.tensor(node.state.get_state_flatten(), dtype=torch.float32)
                predictions = torch.softmax(self.dp_nn(state), dim=0)
                legal = torch.tensor(node.state.get_validity_of_children(), dtype=torch.float32)
                index = torch.argmax(torch.multiply(predictions, legal)).item()
                next_move = node.state.get_children()[index]

            # Apply the action to the node and get back the next node
            node = node.apply_action(next_move)

        # TODO: return the reward of the node given the player using node class
        if node.player == 1:
            return 1
        else:
            return -1

    def calculate_ucb1(self, node: Node) -> float:
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0 and node.parent.player == 1:
            return np.inf
        elif node.visits == 0 and node.parent.player == 2:
            return -np.inf

        elif node.parent.player == 1:
            return self.get_max_value_move(node)
        else:
            return self.get_min_value_move(node)

    def get_max_value_move(self, node: Node) -> float:
        """
        Return the max value move for a given node and child
        """
        return self.q_value(node) + self.u_value(node)

    def get_min_value_move(self, node: Node) -> float:
        """
        Return the min value move for a given node and child
        """
        return self.q_value(node) - self.u_value(node)

    def q_value(self, node: Node) -> float:
        """
        Calculate the Q(s,a) value for a given node
        """
        return node.rewards / node.visits

    def u_value(self, node: Node) -> float:
        """
        Exploration bonus: calculate the U(s,a) value for a given node
        Using upper confidence bound for trees (UCT)
        """
        return self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def select_best_child(self, node: Node) -> Node:
        """
        Select the best child node using UCB1
        """
        ucb1_scores = [self.calculate_ucb1(child) for child in node.children]
        best_idx = np.argmax(
            ucb1_scores) if node.player == 1 else np.argmin(ucb1_scores)
        return node.children[best_idx]

    def node_expansion(self, node: Node) -> Node:
        # Expand node by adding one of its unexpanded children
        # Get the legal moves from the current state
        legal_moves = node.get_legal_moves()

        # Expand the node by creating child nodes for each legal move
        for move in legal_moves:
            node.apply_action(move)

        # Tree policy: return the first child node
        return random.choice(node.children)

    def simulate(self, node: Node) -> int:
        if random.random() < self.sigma:
            return self.default_policy_rollout(node)
        else:
            return self.chritic(node)

    def chritic(self, node: Node):
        # TODO: Use the chritic neural network to simulate a playout from the current node
        pass

    def backpropagate(self, node: Node, reward) -> None:
        # Clear the children of the node generated by the rollout
        node.children = []
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def tree_search(self, node: Node) -> Node:
        # Run while the current node is not a leaf node
        while len(node.children) != 0:
            node = self.select_best_child(node)

        # Test if node is terminal
        if node.is_game_over():
            return node

        # Test if node has been visited before or if it is the root node
        if node.visits != 0 or node == self.root:
            # For each available action from the current state, create a child node and add it to the tree
            return self.node_expansion(node)

        # Return the node to be simulated (rollout)
        return node

    def get_best_move(self) -> Node:
        return max(self.root.children, key=lambda c: c.visits)

    def get_distribution(self):
        total_visits = sum(child.visits for child in self.root.children)
        return self.root.state, [(child.visits / total_visits) for child in self.root.children]

    def search(self, starting_player) -> Node:
        node: Node = self.root
        self.root.player = starting_player

        for _ in range(self.iterations):
            leaf_node = self.tree_search(node)  # Tree policy
            reward = self.simulate(leaf_node)  # Rollout
            self.backpropagate(leaf_node, reward)  # Backpropagation

        # Use the edge (from the root) with the highest visit count as the actual move.
        best_move = self.get_best_move()
        distribution = self.get_distribution()
        return best_move, distribution

    def reset(self) -> None:
        self.root = None
