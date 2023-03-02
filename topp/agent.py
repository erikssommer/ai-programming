import torch
from nn.nn import Actor

# Agent for perticipating in turnament
class Agent:
    def __init__(self, network_path, filename):
        self.player = filename # Naming the player the same as the network for clarity
        self.win = 0
        self.loss = 0
        self.anet = Actor(10, 10, 64)
        self.anet.load_state_dict(torch.load(network_path + filename))
        self.anet.eval()

    def __str__(self):
        return self.player

    def __repr__(self):
        return self.player

    # Play a round of the turnament
    def make_move(self, game):
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

    # Reset the agent's score
    def reset_score(self):
        self.score = 0

    # Get the agent's score
    def get_score(self):
        return self.score

    # Get the agent's name
    def get_player(self):
        return self.player

    