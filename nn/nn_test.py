import copy
import random

from tqdm import tqdm

from managers.state_manager import StateManager
from nn.on_policy import OnPolicy
from ui.ui_init import ui_setup
from utility.read_config import config
import torch


def test_nn():
    ui = ui_setup()

    for _ in tqdm(range(100)):
        state_manager: StateManager = StateManager.create_state_manager()
        game_state = state_manager.get_game_state()

        ui.board = game_state

        ui.draw_board()

        ann = OnPolicy(states=config.board_size ** 2 + 1,
                 actions=config.board_size ** 2)

        ann.load_state_dict(torch.load(f'./nn_models/anet100_hex.pt'))

        while not state_manager.is_game_over():

            input("Press Enter to continue...")

            action = ann.best_action(state_manager)
            print("action", action)

            input("Press Enter to continue...")

            state_manager.apply_action_self(action)
            ui.draw_board()

