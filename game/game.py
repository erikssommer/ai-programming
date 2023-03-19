from abc import ABC, abstractmethod

class Game(ABC):

    @abstractmethod
    def get_state_flatten(self):
        pass

    @abstractmethod
    def perform_action(self, state):
        pass

    @abstractmethod
    def apply_action(self, input_action):
        pass

    @abstractmethod
    def apply_action_self(self, input_action):
        pass

    @abstractmethod
    def validate_action(self, input_action):
        pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_legal_actions(self):
        pass

    @abstractmethod
    def get_validity_of_children(self):
        pass

    @abstractmethod
    def get_winner(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def is_game_over(self):
        pass

    @abstractmethod
    def is_game_over_with_player(self):
        pass
