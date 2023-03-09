import torch
from nn.on_policy import OnPolicy
from utility.read_config import config

# Agent for perticipating in turnament
class Agent:
    def __init__(self, network_path, filename):
        self.name = filename # Naming the player the same as the network for clarity
        self.win = 0
        self.loss = 0
        self.draw = 0
        self.anet = OnPolicy(config.board_size**2, config.board_size**2, 64, optimizer=config.optimizer, activation=config.activation, lr=config.lr)
        self.anet.load_state_dict(torch.load(network_path + filename))
        self.anet.eval()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    # Play a round of the turnament
    def choose_action(self, game):
        value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
        argmax = torch.multiply(torch.softmax(self.anet(value), dim=0), torch.tensor(game.get_validity_of_children())).argmax().item()
        action = game.get_children()[argmax]
        return action

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
        torch.save(self.anet.state_dict(), path + self.name)

    