import random
from time import sleep
import torch

import pygame.display
from nn.on_policy import OnPolicy

from ui.hex import HexUI

from game.hex import HexGame

from utility.read_config import config


def main():
    won = 0
    lost = 0
    draw = 0
    ui = HexUI(config.board_size)
    for _ in range(20):
        game = HexGame(dim=config.board_size)
        ui.board = game.game_state
        ui.draw_board()

        nn = OnPolicy(config.board_size**2, config.board_size**2, 64)
        nn.load_state_dict(torch.load("nn_models/anet80.pt"))

        while not game.is_game_over():
            if game.player == 2:
                game.apply_action_self(random.choice(game.get_legal_actions()))
            else:
                value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
                argmax = torch.multiply(nn(value), torch.tensor(game.get_validity_of_children())).argmax().item()
                action = game.get_children()[argmax]
                game.apply_action_self(action)

            ui.draw_board()
            sleep(0.5)

        # print nn won if game.winner == 1 won, lost if game.winner == 2 won and draw game.winner == 0
        print(f"{'NN won' if game.get_winner() == 1 else 'NN lost' if game.get_winner() == 2 else 'draw'}!")

        if game.get_winner() == 1:
            won += 1
        elif game.get_winner() == 2:
            lost += 1
        else:
            draw += 1

        sleep(1)
    print(f"NN won {won} times, lost {lost} times and drew {draw} times.")


if __name__ == "__main__":
    main()
