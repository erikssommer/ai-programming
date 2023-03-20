from nn.on_policy import OnPolicy
from managers.state_manager import StateManager
from utility.read_config import config

# Agent for perticipating in turnament


class Agent:
    def __init__(self, network_path, filename):
        self.name = filename  # Naming the player the same as the network for clarity
        self.win = 0
        self.loss = 0
        self.draw = 0
        self.anet = self.on_policy_setup(network_path, filename)

    def on_policy_setup(self, network_path, filename):
        if config.game == "hex":
            return OnPolicy(states=config.board_size**2, actions=config.board_size**2, load=True, model_path=network_path + filename)
        elif config.game == "nim":
            return OnPolicy(states=sum(range(config.nr_of_piles + 1)), actions=sum(range(config.nr_of_piles + 1)), load=True, model_path=network_path + filename)
        else:
            raise Exception("Game not supported")

    # Play a round of the turnament
    def choose_action(self, state: StateManager):
        return self.anet.best_action(state)

    # Add a win
    def add_win(self):
        self.win += 1

    # Add a loss
    def add_loss(self):
        self.loss += 1

    # Add a draw
    def add_draw(self):
        self.draw += 1

    # Reset the agent's score
    def reset_score(self):
        self.score = 0

    # Get the agent's score
    def get_score(self):
        return self.score

    # Get the agent's name
    def get_name(self):
        return self.name

    def save_model(self, path):
        self.anet.save(path + self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
