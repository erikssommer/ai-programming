from ActorClient import ActorClient
from MyHexActor import MyHexActor
from dotenv import load_dotenv
from managers.state_manager import StateManager
import os

load_dotenv()

# Set the auth token and whether to qualify
auth_token = os.getenv('AUTH_TOKEN')
qualify = False

# Create the actor for the client to use
actor = MyHexActor()

state_manager = StateManager.create_state_manager(oht=True)

# Create the client
class MyClient(ActorClient):

    def handle_game_start(self, start_player):
        state_manager.set_player(start_player)
    
    # Choosing an action based on the state
    def handle_get_action(self, state):
        state_manager.set_game_state(state)
        row, col = actor.get_action(state_manager)
        return int(row), int(col)

# Run the client
if __name__ == '__main__':
    client = MyClient(auth=auth_token, qualify=qualify)
    client.run()
