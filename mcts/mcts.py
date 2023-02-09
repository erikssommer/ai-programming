import random
import numpy as np
from node import Node


class MCTS:
    def __init__(self, state, nnet, iterations=1000):
        self.iterations = iterations
        self.root = Node(state)
        self.nnet = nnet
        self.c = 1.4

    def _random_playout(self, node: Node):
        """
        Default rollout policy: play randomly until the end of the game
        """
        current_state = node.state
        while not current_state.is_terminal():
            state = random.choice(current_state.get_legal_actions())
        return state.get_reward()
    
    def _calculate_ucb1(self, node: Node, child: Node):
        """
        Calculate UCB1 value for a given node and child
        """
        if child.visits == 0:
            return np.inf
        else:
            return child.rewards + self.c * np.sqrt(np.log(node.visits) / (1 + child.visits))
    
    def _select_best_child(self, node: Node):
        # Select child with highest UCB1 value
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            ucb1 = self._calculate_ucb1(node, child)
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        return best_child
    
    def _expand(self, node: Node):
        # Expand node by adding one of its unexpanded children
        unexpanded_children = [child for child in node.state.get_children() if child not in [c.state for c in node.children]]
        if unexpanded_children:
            child_state = random.choice(unexpanded_children)
            return node.add_child(child_state)
        return None
    
    def _simulate(self, node: Node, epsilon=2.0, player=1):
        if random.random() < epsilon:
            return self._random_playout(node)
        else:
            return self._chritic(node, player)
    
    def _chritic(self, node: Node, player):
        # Use the neural network to simulate a playout from the current node
        state = node.state
        split_state = np.concatenate(([player], [int(i) for i in state.split()]))
        preds = self.nnet.predict(np.array([split_state]))
        return self.nnet.best_action(preds[0])
    
    def _backpropagate(self, node: Node, reward):
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def search(self, player) -> Node:
        for _ in range(self.iterations):
            node = self.root
            while node.children:
                node = self._select_best_child(node)
            child = self._expand(node)
            if child is not None:
                node = child
            reward = self._simulate(node, player)
            self._backpropagate(node, reward)
        return max(self.root.children, key=lambda c: c.visits)
