import numpy as np
import random
from game.game import Game

from copy import deepcopy

# The game of Hex


class HexGame(Game):

    def __init__(self, game_state=None, root=None, dim=7):
        self.player = 1
        self.game_state = game_state if game_state is not None else np.zeros(
            (dim, dim))
        self.dim = dim

    def get_state_flatten(self):
        return list(self.game_state.flatten())

    def perform_action(self, state):
        self.game_state = state.game_state
        self.player = self.player % 2 + 1

    def apply_action(self, input_action):
        if not self.validate_action(input_action):
            print("Invalid action")
            raise Exception("Invalid action")

        new_state = deepcopy(self.game_state)
        new_state[input_action[0]][input_action[1]] = self.player
        new_game = HexGame(new_state, dim=self.dim)
        new_game.player = self.player % 2 + 1
        new_game.action = input_action
        return new_game

    def apply_action_self(self, input_action):
        if not self.validate_action(input_action):
            print("Invalid action")
            raise Exception("Invalid action")

        self.game_state[input_action[0]][input_action[1]] = self.player
        self.player = self.player % 2 + 1

    def validate_action(self, input_action):
        if self.game_state[input_action[0]][input_action[1]] != 0:
            return False
        return True

    def get_children(self):
        children = []
        for i in range(self.dim):
            for j in range(self.dim):
                children.append((i, j))
        return children

    def get_legal_actions(self):
        valid_actions = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.game_state[i][j] == 0:
                    valid_actions.append((i, j))
        return valid_actions

    def get_validity_of_children(self):
        value = np.ones((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                if self.game_state[i][j] != 0:
                    value[i][j] = 0
        return list(value.flatten())

    def get_winner(self):
        """
        :return: player
        """

        over, winner = self.is_game_over_with_player()

        if over:
            return winner
        else:
            return 0

    def get_reward(self):
        """
        :return: reward
        """
        over, winner = self.is_game_over_with_player()

        if over:
            if winner == 1:
                return 1
            elif winner == 2:
                return -1
            else:
                return 0
        else:
            return 0

    def is_game_over(self):
        done, winner = self.is_game_over_with_player()

        return done

    def is_game_over_with_player(self):
        # check if player 1 has a path from top to bottom
        for i in range(self.dim):
            if self.game_state[0][i] == 1:
                if self.check_path((0, i), 1):
                    return True, 1

        # check if player 2 has a path from left to right
        for i in range(self.dim):
            if self.game_state[i][0] == 2:
                if self.check_path((i, 0), 2):
                    return True, 2

        # check if there are no winning states left
        if len(self.get_legal_actions()) == 1 and self.apply_action(self.get_legal_actions()[0]).is_game_over():
            return self.apply_action(self.get_legal_actions()[0]).is_game_over_with_player()

        # if no 0s left, game is over
        for i in range(self.dim):
            for j in range(self.dim):
                if self.game_state[i][j] == 0:
                    return False, 0

        return True, 0

    def check_path(self, start, player):
        visited = np.zeros((self.dim, self.dim))
        return self.check_path_helper(start, player, visited)

    def check_path_helper(self, start, player, visited):
        if start[0] < 0 or start[0] >= self.dim or start[1] < 0 or start[1] >= self.dim:
            return False

        if self.game_state[start[0]][start[1]] != player:
            return False

        if visited[start[0]][start[1]] == 1:
            return False

        visited[start[0]][start[1]] = 1

        if player == 1:

            if start[0] == self.dim - 1:
                return True

            else:
                return self.or_child(start, player, visited)

        else:

            if start[1] == self.dim - 1:
                return True

            else:
                return self.or_child(start, player, visited)

    def or_child(self, start, player, visited):
        return self.check_path_helper((start[0] - 1, start[1] + 1), player, visited) or \
            self.check_path_helper((start[0] - 1, start[1]), player, visited) or \
            self.check_path_helper((start[0], start[1] + 1), player, visited) or \
            self.check_path_helper((start[0], start[1] - 1), player, visited) or \
            self.check_path_helper((start[0] + 1, start[1] - 1), player, visited) or \
            self.check_path_helper((start[0] + 1, start[1]), player, visited)


def demo():
    game = HexGame(initial=True)

    while not game.is_game_over():
        game.apply_action_self(random.choice(game.get_legal_actions()))

    print(f"Player {game.get_winner()} wins!")
