from nn.on_policy import OnPolicy

# Agent for perticipating in turnament
class Agent:
    def __init__(self, network_path, filename):
        self.name = filename # Naming the player the same as the network for clarity

        self.player_1_win = 0
        self.player_2_win = 0

        self.player_1_loss = 0
        self.player_2_loss = 0

        self.win = 0
        self.loss = 0
        self.draw = 0
        self.anet = OnPolicy(load=True, model_path=network_path + filename)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    # Play a round of the turnament
    def choose_action(self, game):
        return self.anet.best_action(game)

    # Add a win
    def add_win(self, player):
        self.win += 1

        if player == 1:
            self.player_1_win += 1
        else:
            self.player_2_win += 1

    # Add a loss
    def add_loss(self, player):
        self.loss += 1

        if player == 1:
            self.player_1_loss += 1
        else:
            self.player_2_loss += 1

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

    