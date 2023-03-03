import torch
import os
from buffers.rbuf import RBUF
from game.hex import HexGame
from game.nim import NimGame
from mcts.mcts import MCTS
from utility.read_config import config
from nn.nn import OnPolicy
from tqdm.auto import tqdm
from topp.topp import TOPP
from utility.timer import Timer


def train_models():
    # i_s = save interval for ANET (the actor network) parameters
    save_interval = config.nr_of_games // config.nr_of_anets

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(config.rbuf_size)

    # Randomly initialize parameters (weights and biases) of ANET
    ann = OnPolicy(states=config.board_size ** 2, actions=config.board_size ** 2, hidden_size=64)
    #ann = Actor(states=sum(range(config.nr_of_piles + 1)), actions=sum(range(config.nr_of_piles + 1)), hidden_size=64)
    # Setting the activation of default policy network and critic network
    epsilon = config.epsilon
    sigma = config.sigma

    acc = 0

    starting_player = 1

    # For g_a in number actual games
    for g_a in tqdm(range(config.nr_of_games)):
        # Initialize the actual game board (B_a) to an empty board.
        #game = NimGame(NimGame.generate_state(config.nr_of_piles), initial=True)
        game = HexGame(initial=True, dim=config.board_size)
        game.player = starting_player

        # s_init ← starting board state
        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        tree = MCTS(game.root_node, epsilon, sigma, config.nr_of_simulations, config.c, dp_nn=ann)

        # For testing purposes
        node = tree.root

        # While B_a not in a final state:
        while not game.is_game_over():
            # Initialize Monte Carlo game board (Bmc) to same state as current game board state (B_a)
            best_move_node, distribution = tree.search(game.player)

            # Add case (root, D) to RBUF
            rbuf.add_case((tree.root, distribution))

            # Choose actual move (a*) based on D
            # Perform a* on root to produce successor state s*
            game.perform_action(best_move_node.state)

            # Update Ba to s*
            # In MCT, retain subtree rooted at s*; discard everything else.
            # root ← s*
            tree.root = best_move_node

        if config.visualize_tree:
            graph = node.visualize_tree()
            graph.render('./visualization/images/tree', view=True)

        # Print the result of the game
        #print(f"Player {str(game.get_winner())} wins!")
        #time.sleep(2)

        if game.get_winner() == 1:
            acc += 1

        starting_player = 3 - starting_player

        # Resetting the tree
        tree.reset()

        # Updating sigma and epsilon
        epsilon = epsilon * config.epsilon_decay
        sigma = sigma * config.sigma_decay

        # Train ANET on a random minibatch of cases from RBUF
        ann.train_step(rbuf.get(128))

        # if g_a modulo is == 0:
        if g_a % save_interval == 0:
            # Save early ANET’s model for later use in tournament play.
            torch.save(ann.state_dict(), f'./nn_models/anet{g_a}.pt')

    # Save final ANET’s model for later use in tournament play.
    torch.save(ann.state_dict(), f'./nn_models/anet{config.nr_of_games}.pt')

    print(f"Player 1 won {acc} of {config.nr_of_games} games.")

def play_topp():
    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = TOPP(config.nr_of_anets, config.nr_of_topp_games)

    # Add the agents to the tournament
    topp.add_agents()

    # Run the tournament
    topp.run_turnament()

    # Get the results
    topp.get_results()

def setup():
    # Create the folder for models if not already existing
    if not os.path.exists('./nn_models'):
        os.makedirs('./nn_models')


if __name__ == "__main__":
    setup()
    if config.train:
        timer = Timer()
        timer.start_timer()
        train_models()
        timer.end_timer()
    if config.topp:
        play_topp()
