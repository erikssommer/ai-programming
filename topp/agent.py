import torch

# Agent for perticipating in turnament
class Agent:
    def __init__(self, player, network_path):
        self.player = player
        self.skills = torch.load(network_path)
        self.score = 0

    def __str__(self):
        return self.player

    def __repr__(self):
        return self.player

    # Play a round of the turnament
    def play(self, other):
        return self.strategy.play(self, other)

    # Add a point to the agent's score
    def add_point(self):
        self.score += 1

    # Reset the agent's score
    def reset_score(self):
        self.score = 0

    # Get the agent's score
    def get_score(self):
        return self.score

    # Get the agent's name
    def get_player(self):
        return self.player

    # Get the agent's strategy
    def get_strategy(self):
        return self.strategy

    # Set the agent's strategy
    def set_strategy(self, strategy):
        self.strategy = strategy
    