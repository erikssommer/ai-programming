import copy
import random

from tqdm import tqdm

from managers.state_manager import StateManager
from mcts.mcts import MCTS
from mcts.node import Node
from nn.on_policy import OnPolicy
from utility.read_config import config


def test_mcts():
    victories = 0

    mcts_player = 1

    for _ in tqdm(range(100)):
        state_manager: StateManager = StateManager.create_state_manager()
        game_state = copy.deepcopy(state_manager.get_game_state())

        ann = OnPolicy(states=config.board_size ** 2 + 1,
                 actions=config.board_size ** 2)

        tree = MCTS(game_state, 1, 1, 100, 1, dp_nn=ann)

        while not state_manager.is_game_over():

            if state_manager.get_player() == mcts_player:
                best_move_node, distribution = tree.search(
                    state_manager.get_player())

                state_manager.perform_action(best_move_node.state)

            else:
                actions = state_manager.get_legal_actions()
                action = random.choice(actions)

                state_manager.apply_action_self(action)

                node = Node(None, game_state=state_manager.get_game_state(), root_node=True)
                node.state.state.player = state_manager.get_player()
                tree.root = node

        if state_manager.get_winner() == mcts_player:
            victories += 1

        mcts_player = 3 - mcts_player

    print(f"Victories: {victories}, {victories/100*100}%")
