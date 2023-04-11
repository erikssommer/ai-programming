import random
from managers.state_manager import StateManager
from mcts.mcts import MCTS
import unittest
import copy
import numpy as np

class TestMCTS(unittest.TestCase):

    def test_mcts(self):
        victories = 0
        games = 10
        rollouts = 100
        mcts_player = 1

        for _ in range(games):
            state_manager: StateManager = StateManager.create_state_manager()
            game_state = copy.deepcopy(state_manager.get_game_state())

            tree = MCTS(game_state, 1, 1, rollouts, 1, dp_nn=None)

            while not state_manager.is_game_over():
                if state_manager.get_player() == mcts_player:
                    _, _, game_state, _ = tree.search(state_manager.get_player())
                    state_manager.perform_action(np.array(game_state))
                else:
                    actions = state_manager.get_legal_actions()
                    action = random.choice(actions)
                    state_manager.apply_action_self(action)
                    tree = MCTS(state_manager.get_game_state(), 1, 1, rollouts, 1)
                    tree.root.state.state.player = state_manager.get_player()

            if state_manager.get_winner() == mcts_player:
                victories += 1

            mcts_player = 3 - mcts_player

        # Calculate the win probability
        win_probability = victories / games

        # Assert the results
        self.assertTrue(win_probability > 0.9)

if __name__ == '__main__':
    print("Running tests...")
    unittest.main()
