import torch
from buffers.rbuf import RBUF
from game.hex import HexGame
from mcts.mcts import MCTS
from utility.read_config import config
import time
from datetime import datetime
from nn.nn import Actor
from tqdm.auto import tqdm
from topp.topp import TOPP


def train_models():
    # i_s = save interval for ANET (the actor network) parameters
    save_interval = config.nr_of_games // config.nr_of_anets

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(config.rbuf_size)

    # Randomly initialize parameters (weights and biases) of ANET
    ann = Actor(states=config.board_size**2, actions=config.board_size**2, hidden_size=64)

    # Setting the activation of default policy network and critic network
    epsilon = config.epsilon
    sigma = config.sigma

    acc = 0

    # For g_a in number actual games
    for g_a in tqdm(range(config.nr_of_games)):
        # Initialize the actual game board (B_a) to an empty board.
        #game = NimGame(NimGame.generate_state(config.nr_of_piles), initial=True)
        game = HexGame(initial=True, dim=config.board_size)

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

            #clear_rewards(tree.root)

        if config.visualize_tree:
            graph = node.visualize_tree()
            graph.render('./visualization/images/tree', view=True)

        # Print the result of the game
        #print(f"Player {str(game.get_winner())} wins!")
        #time.sleep(2)

        if game.get_winner() == 1:
            acc += 1

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


if __name__ == "__main__":
    if config.train:
        # Format the current time
        FMT = '%H:%M:%S'
        start_datetime = time.strftime(FMT)
        train_models()
        end_datetime = time.strftime(FMT)
        # Calculate the time difference
        total_datetime = datetime.strptime(end_datetime, FMT) - datetime.strptime(start_datetime, FMT)
        print(f"Played {config.nr_of_games} games with and {config.nr_of_simulations} simulations per move")
        print(f"Started: {start_datetime}\nFinished: {end_datetime}\nTotal: {total_datetime}")
    elif config.topp:
        play_topp()
