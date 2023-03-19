from tqdm import tqdm
from buffers.rbuf import RBUF
from mcts.mcts import MCTS
from nn.on_policy import OnPolicy
from utility.read_config import config
from managers.state_manager import StateManager

# The Reinforcement Learning (RL) Algorithm

def rl():
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

    # Set the number of simulations and c constant
    simulations = config.simulations
    c = config.c

    acc = 0

    starting_player = 1

    # Saving the initial anet model
    ann.save(f'./nn_models/anet0_{config.game}.pt')

    # For g_a in number actual games
    for episode in tqdm(range(config.episodes)):
        # Initialize the actual game board (B_a) to an empty board.
        state_manager: StateManager = StateManager.create_state_manager()
        
        state_manager.set_player(starting_player)

        game_state = state_manager.get_game_state()

        # s_init ← starting board state
        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        tree = MCTS(game_state, epsilon, sigma,
                    simulations, c, dp_nn=ann)

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