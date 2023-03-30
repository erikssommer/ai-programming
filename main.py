import os
from utility.read_config import config
from topp.topp import TOPP
from utility.timer import Timer
from rl.reinforcement_learning import RL

# Main file for training and playing the Tournament of Progressive Policies (TOPP)


def train_models():
    # Initialize the reinforcement learning
    rl = RL()

    # Start the timer
    timer = Timer()
    timer.start_timer()

    # Train the models
    rl.learn()

    # End the timer
    timer.end_timer()


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
    if not os.path.exists('./models'):
        os.makedirs('./models')


def delete_models():
    # Delete folder in case it already exists
    if os.path.exists('./models/best_model'):
        for file in os.listdir('./models/best_model'):
            os.remove(os.path.join('./models/best_model', file))
        os.rmdir('./models/best_model')
    # Delete all models in the folder
    for file in os.listdir('./models'):
        os.remove(os.path.join('./models', file))


if __name__ == "__main__":
    setup()
    if config.train:
        delete_models()
        train_models()
    if config.topp:
        play_topp()
