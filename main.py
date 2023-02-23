import os
from dotenv import load_dotenv
from stats.rbuf import RBUF
from nim.nim import NimGame
from mcts.mcts import MCTS
from visualization.visualize_tree import VisualizeTree
from read_config import config

from nn.nn import Actor

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import itertools

def main():
    # i_s = save interval for ANET (the actor network) parameters
    save_interval = config.nr_of_games // config.nr_of_anets

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(config.rbuf_size)

    # TODO: Randomly initialize parameters (weights and biases) of ANET
    ann = Actor(states=10, actions=10, hidden_size=64)

    # For g_a in number actual games
    for g_a in tqdm(range(config.nr_of_games)):
        # Initialize the actual game board (B_a) to an empty board.
        game = NimGame(NimGame.generate_state(config.nr_of_piles), initial=True)

        # TODO: s_init ← starting board state

        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init

        tree = MCTS(game.root_node, config.epsilon, config.sigma, config.nr_of_simulations, config.c, dp_nn=ann)

        # For testing purposes
        node = tree.root

        # While B_a not in a final state:
        while not game.is_game_over():
            # Initialize Monte Carlo game board (Bmc) to same state as current game board state (B_a)
            # tree.root = game.get_state() # TODO: method needed
            best_move_node, distribution = tree.search(game.player)

            # Add case (root, D) to RBUF
            rbuf.add_case((tree.root, distribution, best_move_node.state))

            # Choose actual move (a*) based on D
            # Done in mcts.py

            # TODO: Perform a* on root to produce successor state s*
            game.perform_action(best_move_node.state)

            # TODO: Update Ba to s*

            # In MCT, retain subtree rooted at s*; discard everything else.
            # root ← s*
            tree.root = best_move_node

        if config.visualize_tree:
            VisualizeTree(node).visualize_tree()

        # Print the result of the game
        print(f"Player {str(game.get_winner())} wins!")

        # Resetting the tree
        tree.reset()

        # Updating sigma and epsilon
        tree.sigma = tree.sigma * config.sigma_decay
        tree.epsilon = tree.epsilon * config.epsilon_decay

        # TODO: Train ANET on a random minibatch of cases from RBUF

        ann.train_step(rbuf.get(128))

        # if g_a modulo is == 0:
        if g_a > 1 and g_a % save_interval == 0:
            # TODO: Save ANET’s current parameters for later use in tournament play.
            pass

        print(f"Player {str(game.get_winner())} wins!")

if __name__ == "__main__":
    main()
