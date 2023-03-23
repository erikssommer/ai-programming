from tqdm import tqdm
from buffers.rbuf import RBUF
from mcts.mcts import MCTS
from nn.on_policy import OnPolicy
from utility.read_config import config
from managers.state_manager import StateManager
from ui.ui_init import ui_setup

# The Reinforcement Learning (RL) Algorithm


class RL:

    def on_policy_setup(self):
        if config.game == "hex":
            return OnPolicy(states=config.board_size**2 + 1,
                            actions=config.board_size**2)
        elif config.game == "nim":
            return OnPolicy(states=sum(range(config.nr_of_piles + 1)) + 1,
                            actions=sum(range(config.nr_of_piles + 1)))
        else:
            raise Exception("Game not supported")

    def learn(self):
        if config.train_ui:
            ui = ui_setup()

        # i_s = save interval for ANET (the actor network) parameters
        save_interval = config.episodes // config.nr_of_anets

        # Clear Replay Buffer (RBUF)
        rbuf = RBUF(config.rbuf_size)

        # Randomly initialize parameters (weights and biases) of ANET
        ann = self.on_policy_setup()

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

            # Set the starting player
            state_manager.set_player(starting_player)

            # s_init ← starting board state
            game_state = state_manager.get_game_state()

            # If UI for training is enabled, draw the board
            if config.train_ui:
                ui.board = game_state
                ui.draw_board()

            # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
            tree = MCTS(game_state, epsilon, sigma, simulations, c, dp_nn=ann)

            # For testing purposes
            node = tree.root

            # While B_a not in a final state:
            while not state_manager.is_game_over():
                # Initialize Monte Carlo game board (Bmc) to same state as current game board state (B_a)
                # Running the MCTS algorithm on Bmc will produce a distribution over actions
                best_move_node, distribution = tree.search(
                    state_manager.get_player())

                # Add case (root, D) to RBUF
                rbuf.add_case((tree.root, distribution))

                # Choose actual move (a*) based on D
                # Perform a* on root to produce successor state s*
                state_manager.perform_action(best_move_node.state)

                # If UI for training is enabled, draw the current board
                if config.train_ui:
                    ui.board = state_manager.get_game_state()
                    ui.draw_board()

                # Update Ba to s*
                # In MCT, retain subtree rooted at s*; discard everything else.
                # root ← s*
                tree.root = best_move_node

            # Visualize the tree generated by the MCTS
            if config.visualize_tree:
                graph = node.visualize_tree()
                graph.render('./visualization/images/tree', view=True)

            if state_manager.get_winner() == 1:
                acc += 1

            # Switch starting player
            #starting_player = 1 if starting_player == 2 else 2

            # Resetting the tree
            tree.reset()

            # Updating sigma and epsilon
            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay

            # Train ANET on a random minibatch of cases from RBUF
            batch = rbuf.get(config.batch_size)
            for _ in range(10):
                ann.train_step(batch)

            # if g_a modulo is == 0:
            if episode % save_interval == 0 and episode != 0:
                # Save early ANET’s model for later use in tournament play.
                ann.save(f'./nn_models/anet{episode}_{config.game}.pt')

        # Save the final ANET model
        ann.save(f'./nn_models/anet{config.episodes}_{config.game}.pt')

        # Print the number of games won by player 1
        print(f"Player 1 won {acc} of {config.episodes} games.")
