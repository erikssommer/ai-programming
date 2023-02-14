import random
import tkinter as tk
from time import sleep

from copy import deepcopy


class NimGame:
    @staticmethod
    def generate_state(n):
        """
        :param n: number of piles
        :return: the initial state of the game
        """
        return [[1 for _ in range(i)] for i in range(1, n + 1)]

    def __init__(self, game_state, root=None):
        """
        :param game_state: the initial state of the game
        """

        self.player = True

        self.game_state = game_state

        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root

        self.root.title("nim")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.labels = []

    def apply_action(self, input_action):
        """
        :param input_action: the action to be applied to the current state (pile, stones)
        :return: the new state after applying the action
        """

        if not self.validate_action(input_action):
            print("Invalid action")
            raise Exception("Invalid action")
        """
        start = deepcopy(self.game_state)

        for i in range(input_action[1]):
            # set input_action[1] last 1 to 0
            for j in range(len(self.game_state[input_action[0]])):
                if self.game_state[input_action[0]][-j - 1] == 1:
                    self.game_state[input_action[0]][-j - 1] = 0
                    break

        end = self.game_state

        if start == end:
            print("Start", start)
            print("End", end)
            print("Action", input_action)
            print("Game state", self.game_state)
            raise Exception("Invalid action")
        """
        """
        for i in range(input_action[1]):
            # set input_action[1] last 1 to 0
            for j in range(len(self.game_state[input_action[0]])):
                if self.game_state[input_action[0]][-j - 1] == 1:
                    self.game_state[input_action[0]][-j - 1] = 0
                    break"""

        new_state = deepcopy(self.game_state)
        for i in range(input_action[1]):
            # set action[1] last 1 to 0
            for j in range(len(self.game_state[input_action[0]])):
                if new_state[input_action[0]][-j - 1] == 1:
                    new_state[input_action[0]][-j - 1] = 0
                    break

        self.player = not self.player

        return new_state

    def get_state(self):
        return self.game_state

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

        print(f"Player {1 if self.player else 2}, Choose a pile and the number of stones to remove from it.")
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

    def print_piles(self):
        """
        Print the piles of the game state
        GUI used is tkinter
        """

        for widget in self.root.winfo_children():
            widget.destroy()

        if not self.is_game_over():

            label = tk.Label(
                self.root,
                text=f"Player: {self.get_player()}",
                anchor='w'
            )
            label.pack(fill='both')

            for index, value in enumerate(self.game_state):
                label = tk.Label(
                    self.root,
                    text=f"Pile: {index}, stones: {' '.join([f'O' for _ in range(len(list(filter(lambda x: (x == 1), value))))])} \n",
                    anchor='w'
                )

                label.pack(fill='both')

        else:
            label = tk.Label(
                self.root,
                text=f"Player {self.get_player()} wins!",
                anchor='w'
            )
            label.pack(fill='both')
        self.root.update()

    def get_player(self):
        """
        :return: the player who has to play
        """
        return 1 if self.player else 2

    def reward(self):
        if self.is_game_over():
            return 1
        else:
            return 0

    def __str__(self):
        return str(self.game_state)


def flatten(input_list: list) -> list:
    """
    Flatten a list of lists, only works for 2d lists
    :param input_list:
    :return: makes a list of lists into a list
    """
    return [item for sublist in input_list for item in sublist]


def demo():
    """
    Demo of the game
    Two random players play the game
    :return:
    """

    root = tk.Tk()
    game = NimGame(NimGame.generate_state(4), root)

    while not game.is_game_over():
        game.print_piles()
        actions = game.get_legal_actions()
        action = actions[random.randint(0, len(actions) - 1)]
        game = NimGame(game.apply_action(action), root)
        #print(actions)
        print(game.game_state)
        sleep(1)
    game.print_piles()
    sleep(5)
    print(f"You {'won' if game.get_winner() else 'lost'}!")


if __name__ == '__main__':
    demo()


