from time import sleep
import tkinter as tk

from utility.read_config import config

from game.hex import HexGame
import random
from nn.nn import Actor
import torch

def play_game(with_root: bool = False):
    anet = Actor(config.board_size**2, config.board_size**2, 64)
    anet.load_state_dict(torch.load('./nn_models/anet100.pt'))
    anet.eval()

    won = 0
    nr_games = 100
    starting_player = 1

    for _ in range(nr_games):
        if with_root:
            root = tk.Tk()
            #game = NimGame(NimGame.generate_state(7), root=root)
            game = HexGame(root=root, dim=config.board_size)
        else:
            #game = NimGame(NimGame.generate_state(7))
            game = HexGame(dim=config.board_size)

        last_player = None
        game.player = starting_player

        print("Initial state:")
        print(game.game_state)

        print(f"Starting player: {game.player}")

        while not game.is_game_over():

            if game.player == 1:
                print("NN plays:")
                value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
                argmax = torch.multiply(anet(value), torch.tensor(game.get_validity_of_children())).argmax().item()
                action = game.get_children()[argmax]
                game.apply_action_self(action)
                print(game.game_state)
            else:
                print("Random plays:")
                actions = game.get_legal_actions()
                action = actions[random.randint(0, len(actions) - 1)]
                game.apply_action_self(action)
                print(game.game_state)

            if with_root:
                sleep(2)

            last_player = game.player

        winner = game.get_winner()
        # The player gets flippet when move is done so testing for 1 is actually testing for 2
        print(f"NN {'won' if winner == 1 else 'lost'}!")

        if winner == 1:
            won += 1

        starting_player = starting_player % 2 + 1

    print(f"NN won {won} out of {nr_games} games, {won / nr_games * 100}%")


if __name__ == '__main__':
    play_game(with_root=False)
