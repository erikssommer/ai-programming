from time import sleep
import tkinter as tk

from nim.nim import NimGame
import random
from nn.nn import Actor
import torch


def play_game(with_root: bool = False):

    anet = Actor(10, 10, 64)
    anet.load_state_dict(torch.load('./nn_models/anet0.pt'))
    anet.eval()

    print("Starting game")
    won = 0
    nr_games = 100

    for _ in range(nr_games):
        if with_root:
            root = tk.Tk()
            game = NimGame(NimGame.generate_state(4), root=root)
        else:
            game = NimGame(NimGame.generate_state(4))
        
        last_player = None
        while not game.is_game_over():
            if with_root:
                game.print_piles()

            if game.player:
                value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
                argmax = torch.multiply(torch.softmax(anet(value), dim=0)
                                        , torch.tensor(game.get_validity_of_children())).argmax().item()
                action = game.get_children()[argmax]
                game.apply_action_self(action)
                last_player = game.player
            else:
                actions = game.get_legal_actions()
                action = actions[random.randint(0, len(actions) - 1)]
                game.apply_action_self(action)
                last_player = game.player

            if with_root:
                sleep(2)

        print(f"NN {'won' if last_player == 1 else 'lost'}!")
        if last_player == 1:
            won += 1
        if with_root:
            game.print_piles()
            sleep(3)

    print(f"NN won {won} out of {nr_games} games, {won / nr_games * 100}%")


if __name__ == '__main__':
    play_game(with_root=False)


