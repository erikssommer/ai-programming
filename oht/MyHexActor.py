from utility.read_config import config
from nn.on_policy import OnPolicy
from managers.state_manager import StateManager

class MyHexActor():
    def __init__(self):
        self.policy = OnPolicy(states=config.oht_board_size**2 + 1,
                               actions=config.oht_board_size**2,
                               load=True, 
                               model_path='../models/oht/MyHexActor.pt')

    def get_action(self, state):
        return self.policy.get_action(state)

    
    