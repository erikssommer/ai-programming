from oht.ActorClient import ActorClient
from oht.MyHexActor import MyHexActor
from dotenv import load_dotenv
import os

load_dotenv()

# Set the auth token and whether to qualify
auth_token = os.getenv('AUTH_TOKEN')
qualify = False

# Create the actor for the client to use
actor = MyHexActor()

# Create the client
class MyClient(ActorClient):

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        actor.unique_id = unique_id
        actor.series_player_id = series_id
        actor.player_map = player_map
        actor.num_games = num_games # Number of games to be played in the series
        actor.game_params = game_params

        print(f"Unique id: {unique_id}")
        print(f"Series id: {series_id}")
        print(f"Number of games: {num_games}")

    def handle_game_start(self, start_player):
        actor.start_player = start_player

        print(f"Start player: {start_player}")
    
    # Choosing an action based on the state
    def handle_get_action(self, state):
        row, col = actor.get_action(state)
        return int(row), int(col)

# Run the client
if __name__ == '__main__':
    client = MyClient(auth=auth_token, qualify=qualify)
    client.run()
