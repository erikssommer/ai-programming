from time import sleep
from topp.agent import Agent
from ui.hex import HexUI
from utility.read_config import config
import os
import random
import matplotlib.pyplot as plt
from managers.state_manager import StateManager

# The Tournament of Progressive Policies (TOPP)


class TOPP:
    def __init__(self, m, g, ui: bool = False):
        self.m = m+1  # Number of saved anets
        self.g = g  # Number of games to play between each pair of agents
        self.agents: Agent = []
        self.ui = ui

    def add_agents(self):
        policy_path = f"./nn_models/"
        # Get the list of files in the directory
        files = os.listdir(policy_path)

        # Sort the list of files by their modification time
        sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(policy_path, x)))

        for file in sorted_files:
            if file.endswith(".pt"):
                self.agents.append(Agent(policy_path, file))

        # Terminate if there are no agents
        if len(self.agents) == 0:
            print("No agents found, exiting...")
            exit()

    def run_turnament(self):
        if self.ui:
            if config.game == "hex":
                ui = HexUI(config.board_size)
            else:
                ui = None
                print("UI not implemented for this game")

        for i in range(self.m):
            for j in range(i+1, self.m):
                starting_agent = random.choice([i, j])

                # Play a series of G games between agents i and j
                for _ in range(self.g):
                    # Initialize the game
                    state_manager = StateManager.create_state_manager()

                    if self.ui:
                        ui.board = state_manager.get_game_state()

                    current_agent = starting_agent

                    if self.ui:
                        ui.draw_board()

                    # Play the game until it is over
                    while not state_manager.is_game_over():
                        # Get the move from the current player's agent
                        agent: Agent = self.agents[current_agent]
                        action = agent.choose_action(state_manager)

                        # Make the move on the board
                        state_manager.apply_action_self(action)

                        # Swap the current player
                        if current_agent == i:
                            current_agent = j
                        else:
                            current_agent = i

                        if self.ui:
                            ui.draw_board()
                            sleep(0.1)

                    # Record the result of the game
                    winner = state_manager.get_winner()

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
                        self.plot_result(block=False)

                    # Swap the starting player
                    if starting_agent == i:
                        starting_agent = j
                    else:
                        starting_agent = i

    def plot_result(self, block):
        plt.clf()
        plt.ion()
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
        plt.show(block=block)

    def get_results(self):
        agents_result = sorted(self.agents, key=lambda x: x.win, reverse=True)

        for agent in agents_result:
            print(
                f"Agent {agent.name} won {agent.win} times, lost {agent.loss} times and drew {agent.draw} times")
            
        self.save_best_agent(agents_result[0])

        self.plot_result(block=True)
    

    def save_best_agent(self, agent: Agent):
        if not os.path.exists('./nn_models/best_model'):
            os.makedirs('./nn_models/best_model')
        
        agent.save_model('./nn_models/best_model/')




