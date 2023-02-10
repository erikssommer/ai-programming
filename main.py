import os
from dotenv import load_dotenv
from stats.rbuf import RBUF
from nim.nim import NimGame
from mcts.mcts import MCTS

def main():
    # Read environment variables
    rbuf_size = int(os.environ['RBUF_SIZE'])
    nr_of_games = int(os.environ['NR_OF_GAMES'])
    nr_of_anets = int(os.environ['NR_OF_ANETS'])
    nr_of_piles = int(os.environ['NR_OF_PILES'])
    nr_of_simulations = int(os.environ['NR_OF_SIMULATIONS'])
    sigma = float(os.environ['SIGMA'])
    epsilon = float(os.environ['EPSILON'])
    epsilon_decay = float(os.environ['EPSILON_DECAY'])
    sigma_decay = float(os.environ['SIGMA_DECAY'])

    # i_s = save interval for ANET (the actor network) parameters
    save_interval = nr_of_games // nr_of_anets

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(rbuf_size)

    # TODO: Randomly initialize parameters (weights and biases) of ANET

    # For g_a in number actual games
    for g_a in range(nr_of_games):
        # Initialize the actual game board (B_a) to an empty board.
        game = NimGame(nr_of_piles)

        # TODO: s_init ← starting board state

        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        tree = MCTS(game.root_node, epsilon, sigma, nr_of_simulations)

        # While B_a not in a final state:
        while not game.is_over():
            # Initialize Monte Carlo game board (Bmc) to same state as current game board state (B_a)
            tree.root = game.get_state() # TODO: method needed
            best_move_node, distribution = tree.search(game.player)

            # Add case (root, D) to RBUF
            rbuf.add_case((tree.root, distribution, best_move_node.state))

            # Choose actual move (a*) based on D
            # Done in mcts.py

            # TODO: Perform a* on root to produce successor state s*
            game.play(best_move_node.state)

            # TODO: Update Ba to s*

            # In MCT, retain subtree rooted at s*; discard everything else.
            # root ← s*
            tree.root = best_move_node

        # Resetting the tree
        tree.reset()

        # Updating sigma and epsilon
        tree.sigma = tree.sigma * sigma_decay
        tree.epsilon = tree.epsilon * epsilon_decay

        # TODO: Train ANET on a random minibatch of cases from RBUF

        # if g_a modulo is == 0:
        if g_a % save_interval == 0:
            # TODO: Save ANET’s current parameters for later use in tournament play.
            pass

        print(f"Player {str(game.get_winner())} wins!")



if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    main()
