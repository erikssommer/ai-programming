import os
from utility.read_config import config
from topp.topp import TOPP
from utility.timer import Timer
from rl.reinforcement_learning import RL
from tqdm import tqdm
from mcts.mcts import MCTS
from nn.on_policy import OnPolicy
from managers.state_manager import StateManager
import random


# Main file for training and playing the Tournament of Progressive Policies (TOPP)


def train_models():
    # Initialize the reinforcement learning
    rl = RL()
    rl.learn()


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


def test_mcts():
    victories = 0

    for _ in tqdm(range(100)):
        state_manager: StateManager = StateManager.create_state_manager()
        game_state = state_manager.get_game_state()

        ann = OnPolicy(states=config.board_size ** 2 + 1,
                 actions=config.board_size ** 2)

        tree = MCTS(game_state, 1, 1, 1000, 1, dp_nn=ann)

        while not state_manager.is_game_over():

            if state_manager.get_player() == 1:
                best_move_node, distribution = tree.search(state_manager.get_player())

                state_manager.perform_action(best_move_node.state)

            else:
                actions = state_manager.get_legal_actions()
                action = random.choice(actions)

                state_manager.apply_action_self(action)

        if state_manager.get_winner() == 1:
            victories += 1
        
        print(state_manager.get_winner())

    print(victories)


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
    if config.test_mcts:
        test_mcts()
