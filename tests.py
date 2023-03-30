import random
from managers.state_manager import StateManager
from mcts.mcts import MCTS
import unittest

class TestMCTS(unittest.TestCase):

    def test_mcts(self):
        victories = 0
        games = 10
        rollouts = 100

        for _ in range(games):
            state_manager: StateManager = StateManager.create_state_manager()
            game_state = state_manager.get_game_state()

            tree = MCTS(game_state, 1, 1, rollouts, 1, dp_nn=None)

            while not state_manager.is_game_over():
                if state_manager.get_player() == 1:
                    best_move_node, _ = tree.search(state_manager.get_player())
                    state_manager.perform_action(best_move_node.state)
                else:
                    actions = state_manager.get_legal_actions()
                    action = random.choice(actions)
                    state_manager.apply_action_self(action)
                    tree = MCTS(state_manager.get_game_state(), 1, 1, rollouts, 1)

            if state_manager.get_winner() == 1:
                victories += 1

        # Calculate the win probability
        win_probability = victories / games

        # Assert the results
        self.assertTrue(win_probability > 0.9)

if __name__ == '__main__':
    print("Running tests...")
    unittest.main()
