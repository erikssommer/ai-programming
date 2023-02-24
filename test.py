from time import sleep
import tkinter as tk

from nim.nim import NimGame
import random
from nn.nn import Actor
import torch

def play_game(with_root: bool = False):
    anet = Actor(10, 10, 64)
    anet.load_state_dict(torch.load('./nn_models/anet600.pt'))
    anet.eval()

    print("Starting game")
    won = 0
    nr_games = 5000
    starting_player = 1

    for _ in range(nr_games):
        if with_root:
            root = tk.Tk()
            game = NimGame(NimGame.generate_state(4), root=root)
        else:
            game = NimGame(NimGame.generate_state(4))
        
        last_player = None
        game.player = starting_player

        while not game.is_game_over():

            if with_root:
                game.print_piles()

            if game.player == 1:
                value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
                argmax = torch.multiply(torch.softmax(anet(value), dim=0), torch.tensor(game.get_validity_of_children())).argmax().item()
                action = game.get_children()[argmax]
                game.apply_action_self(action)
            else:
                actions = game.get_legal_actions()
                action = actions[random.randint(0, len(actions) - 1)]
                game.apply_action_self(action)

            if with_root:
                sleep(2)
            
            last_player = game.player

        winner = last_player
        # The player gets flippet when move is done so testing for 1 is actually testing for 2
        # print(f"NN {'won' if winner == 2 else 'lost'}!")
        if winner == 2:
            won += 1
        if with_root:
            game.print_piles()
            sleep(3)
        
        starting_player = starting_player % 2 + 1

    print(f"NN won {won} out of {nr_games} games, {won / nr_games * 100}%")


if __name__ == '__main__':
    play_game(with_root=False)


