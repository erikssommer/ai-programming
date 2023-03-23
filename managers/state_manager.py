from game.game import Game
from utility.read_config import config
from game.nim import NimGame
from game.hex import HexGame

# The State Manager


class StateManager:
    def __init__(self, game: Game):
        self.state = game

    def get_player(self):
        return self.state.player

    def set_player(self, player):
        self.state.player = player

    def apply_action(self, action):
        return StateManager(self.state.apply_action(action))

    def apply_action_self(self, action):
        return self.state.apply_action_self(action)

    def perform_action(self, state):
        return self.state.perform_action(state.state)

    def get_legal_actions(self):
        return self.state.get_legal_actions()

    def get_state_flatten(self):
        return self.state.get_state_flatten()

    def get_winner(self):
        return self.state.get_winner()

    def is_game_over(self):
        return self.state.is_game_over()

    def get_children(self):
        return self.state.get_children()

    def get_validity_of_children(self):
        return self.state.get_validity_of_children()

    def get_game_state(self):
        return self.state.game_state

    def get_reward(self):
        return self.state.get_reward()

    # Static class creator for the game
    @staticmethod
    def create_state_manager():
        if config.game == 'nim':
            return StateManager(NimGame(dim=config.nr_of_piles))
        elif config.game == 'hex':
            return StateManager(HexGame(dim=config.board_size))
        else:
            raise ValueError(f"Game {config.game} not supported")
