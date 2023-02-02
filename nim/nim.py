import random
import tkinter as tk


class NimGame:
    def __init__(self, n, solo=False):
        """
        :param n: number of piles
        :param solo: if the game is played solo or two players
        """
        self.n = n
        self.solo = solo
        self.piles = list(range(1, n + 1))
        self.player = True
        self.moves = []

        self.root = tk.Tk()
        self.root.title("nim")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.labels = []

    def apply_action(self, action):
        """
        :param action: the action to be applied to the current state (pile, stones)
        :return: if the game is over
        """
        self.piles[action[0]] -= action[1]
        self.moves.append((self.player, action))
        if self.is_game_over():
            return False
        if self.solo:
            self.make_bot_move()

        self.player = not self.player

        return True

    def make_bot_move(self):
        """
        Make a random move for the bot
        """
        print("Bot is thinking...")
        actions = self.get_children()
        # random move
        action = actions[random.randint(0, len(actions) - 1)]
        self.piles[action[0]] -= action[1]
        self.moves.append(("bot", action))

    def get_children(self):
        """
        :return: the list of all the possible actions for the current state
        """
        actions = []
        for pile in range(self.n):
            for stones in range(1, self.piles[pile] + 1):
                actions.append((pile, stones))
        return actions

    def validate_action(self, action):
        """
        :param action: the action to be validated
        :return: if the action is valid for the current state
        """
        if self.piles[action[0]] >= action[1]:
            return True
        else:
            return False

    def get_action(self):
        """
        :return: the action chosen by the player
        """
        print(f"Player {self.player}, Choose a pile and the number of stones to remove from it.")
        pile = input(f"Pile: ")
        stones = input(f"Stones: ")
        if self.validate_action((int(pile), int(stones))):
            return int(pile), int(stones)
        else:
            print("Invalid action!")
            return self.get_action()

    def is_game_over(self):
        """
        :return: if the game is over
        """
        return all(element == 0 for element in self.piles)

    def get_winner(self):
        """
        :return: the winner of the game
        """
        return not self.player

    def get_piles(self):
        """
        :return: the piles of the game (pile, number of stones)
        """
        return self.piles

    def print_piles(self):
        """
        Print the piles of the game
        """
        for label in self.labels:
            label.destroy()
        self.labels = []

        for index, value in enumerate(self.piles):
            label = tk.Label(
                self.root,
                text=f"Pile: {index}, stones: {' '.join([f'O' for _ in range(value)])} \n",
                anchor='w'
            )

            label.pack(fill='both')

            self.labels.append(label)

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
        return f"Piles: {[f'Pile: {index}, stones: {value}' for index, value in enumerate(self.piles)]}  " \
               f"Player: {str(self.player)}"


def main():
    solo = input("Do you want to play solo? (y/n): ")
    if solo == "y":
        n = input("How many piles do you want to play with? ")
        game = NimGame(int(n), True)
    else:
        n = input("How many piles do you want to play with? ")
        game = NimGame(int(n))
    while not game.is_game_over():
        game.print_piles()
        action = game.get_action()
        game.apply_action(action)

    print(f"Player {str(game.get_winner())} wins!")


if __name__ == "__main__":
    main()
