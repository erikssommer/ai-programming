import random
import tkinter as tk


class NimGame:
    def __init__(self, n, solo=True):
        self.n = n
        self.solo = solo
        self.piles = list(range(1, n + 1))
        self.player = 1
        self.moves = []

        self.root = tk.Tk()
        self.root.title("nim")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.labels = []

    def play(self, action):
        if self.solo:
            if self.piles[action[0]] >= action[1]:
                self.piles[action[0]] -= action[1]
                self.moves.append((self.player, action))
                if self.is_over():
                    return False
                else:
                    self.make_bot_move()
                    return True
        else:
            if self.piles[action[0]] >= action[1]:
                self.piles[action[0]] -= action[1]
                self.moves.append((self.player, action))
                if self.is_over():
                    return False
                self.player = 3 - self.player
                return True
            else:
                return False

    def make_bot_move(self):
        print("Bot is thinking...")
        actions = self.get_actions()
        # random move
        action = actions[random.randint(0, len(actions) - 1)]
        self.piles[action[0]] -= action[1]
        self.moves.append(("bot", action))

    def get_actions(self):
        actions = []
        for pile in range(self.n):
            for stones in range(1, self.piles[pile] + 1):
                actions.append((pile, stones))
        return actions

    def validate_action(self, action):
        if self.piles[action[0]] >= action[1]:
            return True
        else:
            return False

    def get_action(self, ):
        print(f"Player {self.player}, Choose a pile and the number of stones to remove from it.")
        pile = input(f"Pile: ")
        stones = input(f"Stones: ")
        if self.validate_action((int(pile), int(stones))):
            return int(pile), int(stones)
        else:
            print("Invalid action!")
            return self.get_action()

    def is_over(self):
        return all(element == 0 for element in self.piles)

    def get_winner(self):
        if self.solo:
            if self.moves[-1][0] == "bot":
                return 1
            else:
                return "bot"
        return 3 - self.moves[-1][0]

    def get_piles(self):
        return self.piles

    def print_piles(self):

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
        return self.player

    def __str__(self):
        return f"Piles: {[f'Pile: {index}, stones: {value}' for index, value in enumerate(self.piles)]}  " \
               f"Player: {str(self.player)}"
