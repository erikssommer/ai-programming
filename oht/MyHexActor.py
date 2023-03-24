from utility.read_config import config
from nn.on_policy import OnPolicy
from managers.state_manager import StateManager

class MyHexActor():
    def __init__(self):
        self.policy = OnPolicy(states=config.oht_board_size**2 + 1,
                               actions=config.oht_board_size**2,
                               load=True, 
                               model_path="../nn_models/oht/MyHexActor.pt")

    def get_action(self, state: StateManager):
        action = self.policy.best_action(state)
        row, col = action
        return row, col

    
    