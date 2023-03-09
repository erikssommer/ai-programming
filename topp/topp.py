from time import sleep

from game.hex import HexGame
from topp.agent import Agent
from ui.hex import HexUI
from utility.read_config import config
import os
import random
import matplotlib.pyplot as plt
from IPython import display
from IPython.utils import io

# The Tournament of Progressive Policies (TOPP)


class TOPP:
    def __init__(self, m, g, ui: bool = False):
        self.m = m+1  # Number of saved anets
        self.g = g  # Number of games to play between each pair of agents
        self.agents: Agent = []
        self.ui = ui

    def add_agents(self):
        policy_path = f"./nn_models/"
        for file in os.listdir(policy_path):
            if file.endswith(".pt"):
                self.agents.append(Agent(policy_path, file))

        # Terminate if there are no agents
        if len(self.agents) == 0:
            print("No agents found, exiting...")
            exit()

    def run_turnament(self):
        if self.ui:
            ui = HexUI(config.board_size)

        for i in range(self.m):
            for j in range(i+1, self.m):
                starting_agent = random.choice([i, j])

                # Play a series of G games between agents i and j
                for game in range(self.g):
                    # Initialize the game
                    #game = NimGame(NimGame.generate_state(4))
                    game = HexGame(dim=config.board_size)
                    if self.ui:
                        ui.board = game.game_state

                    current_agent = starting_agent

                    if self.ui:
                        ui.draw_board()

                    # Play the game until it is over
                    while not game.is_game_over():
                        # Get the move from the current player's agent
                        agent = self.agents[current_agent]
                        action = agent.choose_action(game)

                        # Make the move on the board
                        game.apply_action_self(action)

                        # Swap the current player
                        if current_agent == i:
                            current_agent = j
                        else:
                            current_agent = i

                        if self.ui:
                            ui.draw_board()
                            sleep(0.1)

                    # Record the result of the game
                    winner = game.get_winner()

                    # Update the agents win/loss/draw
                    if starting_agent == i and winner == 1:
                        self.agents[i].add_win()
                        self.agents[j].add_loss()
                    elif starting_agent == i and winner == 2:
                        self.agents[j].add_win()
                        self.agents[i].add_loss()
                    elif starting_agent == j and winner == 1:
                        self.agents[j].add_win()
                        self.agents[i].add_loss()
                    elif starting_agent == j and winner == 2:
                        self.agents[i].add_win()
                        self.agents[j].add_loss()
                    else:
                        self.agents[i].add_draw()
                        self.agents[j].add_draw()

                    # Update the plot
                    if self.ui:
                        self.update_plot()

                    # Swap the starting player
                    if starting_agent == i:
                        starting_agent = j
                    else:
                        starting_agent = i

    def update_plot(self):
        with io.capture_output() as captured:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            # x is agent name
            x = [agent.name for agent in self.agents]
            # y is number of wins
            y = [agent.win for agent in self.agents]
            # specify colors for each bar
            colors = ['red', 'green', 'blue', 'purple', 'orange',
                      'yellow', 'pink', 'brown', 'black', 'grey']
            plt.bar(x, y, color=colors)
            # Set with of display
            plt.title('Topp Statistics')
            plt.xlabel('Agent wins')
            plt.ylabel('Number of Games')
            plt.show(block=False)

    def get_results(self):
        agents_result = sorted(self.agents, key=lambda x: x.win, reverse=True)

        for agent in agents_result:
            print(
                f"Agent {agent.name} won {agent.win} times, lost {agent.loss} times and drew {agent.draw} times")

        x = [agent.name for agent in self.agents]
        # y is number of wins
        y = [agent.win for agent in self.agents]
        # specify colors for each bar
        colors = ['red', 'green', 'blue', 'purple', 'orange',
                  'yellow', 'pink', 'brown', 'black', 'grey']
        plt.bar(x, y, color=colors)
        # Set with of display
        plt.title('Topp Statistics')
        plt.xlabel('Agent wins')
        plt.ylabel('Number of Games')
        plt.show(block=True)
