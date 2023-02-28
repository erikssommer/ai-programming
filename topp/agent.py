import torch

# Agent for perticipating in turnament
class Agent:
    def __init__(self, network_path, filename):
        self.player = filename # Naming the player the same as the network for clarity
        self.skills = torch.load(network_path + filename)
        self.score = 0

    def __str__(self):
        return self.player

    def __repr__(self):
        return self.player

    # Play a round of the turnament
    def make_move(self, other):
        return self.skills.play(self, other)

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

    