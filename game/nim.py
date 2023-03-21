import random
import tkinter as tk
from time import sleep
from game.game import Game

from copy import deepcopy

# The game of Nim

class NimGame(Game):

    def __init__(self, game_state=None, dim=7):
        """
        :param game_state: the initial state of the game
        """
        self.player = 1
        self.game_state = game_state if game_state is not None else self.generate_state(dim)
        self.dim = dim
    
    def generate_state(self, n):
        """
        :param n: number of piles
        :return: the initial state of the game
        """
        return [[1 for _ in range(i)] for i in range(1, n + 1)]

    def perform_action(self, state):
        self.game_state = state.game_state
        self.player = self.player % 2 + 1

    def apply_action(self, input_action):
        """
        :param input_action: the action to be applied to the current state (pile, stones)
        :return: the new state after applying the action
        """

        if not self.validate_action(input_action):
            print("Invalid action")
            raise Exception("Invalid action")

        new_state = deepcopy(self.game_state)
        for i in range(input_action[1]):
            # set action[1] last 1 to 0
            for j in range(len(self.game_state[input_action[0]])):
                if new_state[input_action[0]][-j - 1] == 1:
                    new_state[input_action[0]][-j - 1] = 0
                    break

        new_game = NimGame(new_state, dim=self.dim)
        new_game.player = self.player % 2 + 1
        new_game.action = input_action

        return new_game

    def apply_action_self(self, input_action):
        """
        :param input_action: the action to be applied to the current state (pile, stones)
        :return: the new state after applying the action
        """

        if not self.validate_action(input_action):
            print("Invalid action")
            raise Exception("Invalid action")
        for i in range(input_action[1]):
            # set input_action[1] last 1 to 0
            for j in range(len(self.game_state[input_action[0]])):
                if self.game_state[input_action[0]][-j - 1] == 1:
                    self.game_state[input_action[0]][-j - 1] = 0
                    break

        self.player = self.player % 2 + 1

    def get_state(self):
        return self.game_state

    def get_state_flatten(self):
        return [item for sublist in self.game_state for item in sublist]

    def get_children(self):
        """
        :return: the list of all the possible actions for the current state
        """

        actions = []
        for index, i in enumerate(self.game_state):
            for j in range(index + 1):
                actions.append((index, j + 1))

        """actions = []
        for pile in range(self.n):
            for stones in range(1, self.piles[pile] + 1):
                actions.append((pile, stones))"""
        return actions

    def get_validity_of_children(self):
        """
        :return: the list of all the possible actions for the current state
        """

        actions = []
        for index, i in enumerate(self.game_state):
            for j in range(index + 1):
                # only append if value at index in list is 1
                if i[j] == 1:
                    actions.append(1)
                else:
                    actions.append(0)

        return actions

    def get_legal_actions(self):
        """
        :return: the list of all the possible actions for the current state
        """

        actions = []
        for index, i in enumerate(self.game_state):
            for j in range(index + 1):
                # only append if value at index in list is 1
                if i[j] == 1:
                    actions.append((index, j + 1))
                else:
                    break

        return actions

    def validate_action(self, action_to_validate):
        """
        :param action_to_validate: the action to be validated
        :return: if the action is valid for the current state
        """
        if len(list(filter(lambda x: (x == 1), self.game_state[action_to_validate[0]]))) < action_to_validate[1]:
            return False
        return True

    def get_player_action(self, ):
        """
        :return: the action chosen by the player
        """

        print(
            f"Player {1 if self.player else 2}, Choose a pile and the number of stones to remove from it.")

        pile = input(f"Pile: ")
        stones = input(f"Stones: ")
        if self.validate_action((int(pile), int(stones))):
            return int(pile), int(stones)
        else:
            print("Invalid action!")
            return self.get_player_action()

    def is_game_over(self):
        """
        :return: if the game is over, all values in the game state are 0
        """
        return all(element == 0 for element in flatten(self.game_state))

    def get_winner(self):
        """
        :return: the winner of the game
        """
        return self.player

    def get_player(self):
        """
        :return: the player who has to play
        """
        return self.player

    def get_reward(self):
        if self.player == 1:
            return 1
        else:
            return -1

    def __str__(self):
        return str(self.game_state)


def flatten(input_list: list) -> list:
    """
    Flatten a list of lists, only works for 2d lists
    :param input_list:
    :return: makes a list of lists into a list
    """
    return [item for sublist in input_list for item in sublist]

