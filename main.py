import os
from dotenv import load_dotenv
from stats.rbuf import RBUF
from nim.nim import NimGame
from mcts.mcts import MCTS

def main():
    # Read environment variables
    rbuf_size = int(os.environ['RBUF_SIZE'])
    nr_of_games = int(os.environ['NR_OF_GAMES'])
    nr_of_piles = int(os.environ['NR_OF_PILES'])
    # TODO: i_s = save interval for ANET (the actor network) parameters

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(rbuf_size)

    # TODO: Randomly initialize parameters (weights and biases) of ANET

    # For g_a in number actual games
    for i in range(nr_of_games):
        # Initialize the actual game board (B_a) to an empty board.
        game = NimGame(nr_of_piles)

        # TODO: s_init ← starting board state

        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        tree = MCTS(game)

        # While B_a not in a final state:
        while not game.is_over():
            # Initialize Monte Carlo game board (Bmc) to same state as root.
            tree.root = game.get_state() # TODO: method needed
            node = tree.search(game.player)
            game.play(node.state)
            game.print_piles()
            action = game.get_action()
            game.play(action)
            game.print_piles()
        print(f"Player {str(game.get_winner())} wins!")



if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    main()
