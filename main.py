import os
from buffers.rbuf import RBUF
from game.hex import HexGame
from game.nim import NimGame
from mcts.mcts import MCTS
from utility.read_config import config
from nn.on_policy import OnPolicy
from tqdm.auto import tqdm
from topp.topp import TOPP
from utility.timer import Timer
from managers.state_manager import StateManager


def train_models():
    # i_s = save interval for ANET (the actor network) parameters
    save_interval = config.episodes // config.nr_of_anets

    # Clear Replay Buffer (RBUF)
    rbuf = RBUF(config.rbuf_size)

    # Randomly initialize parameters (weights and biases) of ANET
    ann = OnPolicy()
    #ann = Actor(states=sum(range(config.nr_of_piles + 1)), actions=sum(range(config.nr_of_piles + 1)), hidden_size=64)

    # Setting the activation of default policy network and critic network
    epsilon = config.epsilon
    sigma = config.sigma

    acc = 0

    starting_player = 1

    # Saving the initial anet model
    ann.save(f'./nn_models/anet0_{config.game}.pt')

    # For g_a in number actual games
    for episode in tqdm(range(config.episodes)):
        # Initialize the actual game board (B_a) to an empty board.
        state_manager: StateManager = StateManager.create_state_manager(config.game)
        
        state_manager.set_player(starting_player)

        # s_init ← starting board state
        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        tree = MCTS(state_manager.get_root_node(), epsilon, sigma,
                    config.simulations, config.c, dp_nn=ann)

        # For testing purposes
        node = tree.root

        # While B_a not in a final state:
        while not state_manager.is_game_over():
            # Initialize Monte Carlo game board (Bmc) to same state as current game board state (B_a)
            best_move_node, distribution = tree.search(state_manager.get_player())

            # Add case (root, D) to RBUF
            rbuf.add_case((tree.root, distribution))

            # Choose actual move (a*) based on D
            # Perform a* on root to produce successor state s*
            state_manager.perform_action(best_move_node.state)

            # Update Ba to s*
            # In MCT, retain subtree rooted at s*; discard everything else.
            # root ← s*
            tree.root = best_move_node

        if config.visualize_tree:
            graph = node.visualize_tree()
            graph.render('./visualization/images/tree', view=True)

        # Print the result of the game
        #print(f"Player {str(game.get_winner())} wins!")
        # time.sleep(2)

        if state_manager.get_winner() == 1:
            acc += 1

        # Switch starting player
        starting_player = 1 if starting_player == 2 else 2

        # Resetting the tree
        tree.reset()

        # Updating sigma and epsilon
        epsilon = epsilon * config.epsilon_decay
        sigma = sigma * config.sigma_decay

        # Train ANET on a random minibatch of cases from RBUF
        ann.train_step(rbuf.get(config.batch_size))

        # if g_a modulo is == 0:
        if episode % save_interval == 0 and episode != 0:
            # Save early ANET’s model for later use in tournament play.
            ann.save(f'./nn_models/anet{episode}_{config.game}.pt')

    # Save the final ANET model
    ann.save(f'./nn_models/anet{config.episodes}_{config.game}.pt')

    print(f"Player 1 won {acc} of {config.episodes} games.")


def play_topp():
    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = TOPP(config.nr_of_anets, config.nr_of_topp_games, ui=config.topp_ui)

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


def delete_models():
    # Delete folder in case it already exists
    if os.path.exists('./nn_models/best_model'):
        for file in os.listdir('./nn_models/best_model'):
            os.remove(os.path.join('./nn_models/best_model', file))
        os.rmdir('./nn_models/best_model')
    # Delete all models in the folder
    for file in os.listdir('./nn_models'):
        os.remove(os.path.join('./nn_models', file))


if __name__ == "__main__":
    setup()
    if config.train:
        delete_models()
        timer = Timer()
        timer.start_timer()
        train_models()
        timer.end_timer()
    if config.topp:
        play_topp()
