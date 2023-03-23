import tkinter as tk
from game.nim import NimGame


class NimUI:
    def __init__(self, nr_of_piles: int, board=None):
        self.board = board
        self.nr_of_piles = nr_of_piles
        self.root = tk.Tk()
        self.game = NimGame(dim=nr_of_piles)

        self.root.title("nim")
        self.root.geometry("500x500")
        self.root.resizable(False, False)

    def draw_board(self):
        """
        Print the piles of the game state
        GUI used is tkinter
        """

        self.game.game_state = self.board

        for widget in self.root.winfo_children():
            widget.destroy()

        if not self.game.is_game_over():

            label = tk.Label(
                self.root,
                text=f"Player: {self.game.get_player()}",
                anchor='w'
            )
            label.pack(fill='both')

            for index, value in enumerate(self.game.game_state):
                label = tk.Label(
                    self.root,
                    text=f"Pile: {index}, stones: {' '.join([f'O' for _ in range(len(list(filter(lambda x: (x == 1), value))))])} \n",
                    anchor='w'
                )

                label.pack(fill='both')

        else:
            label = tk.Label(
                self.root,
                text=f"Player {self.game.get_player()} wins!",
                anchor='w'
            )
            label.pack(fill='both')
        self.root.update()
