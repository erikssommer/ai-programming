from topp.agent import Agent
from nim.nim import NimGame
import numpy as np
import os

# The Tournament of Progressive Policies (TOPP)
class TOPP:
    def __init__(self, m, g):
        self.m = m # Number of saved anets
        self.g = g # Number of games to play between each pair of agents
        self.agents: Agent = []
        self.results = np.zeros((m, m))

    def add_agents(self):
        policy_path = f"./nn_models/"
        for file in os.listdir(policy_path):
            if file.endswith(".pt"):
                print(f"Adding agent {file}")
                self.agents.append(Agent(policy_path, file))
    
    def run_turnament(self):
        for i in range(self.m):
            for j in range(i+1, self.m):
                # Play a series of G games between agents i and j
                for game in range(self.g):
                    # Initialize the game
                    game = NimGame(NimGame.generate_state(4))
                    current_player = 0

                    # Play the game until it is over
                    while not game.is_game_over():
                        # Get the move from the current player's agent
                        agent = self.agents[current_player]
                        state = game.get_state()
                        move = agent.make_move(state)

                        # Make the move on the board
                        game.apply_action_self(move)
                        current_player = 1 - current_player

                    # Record the result of the game
                    winner = game.get_winner()
                    self.results[i, j] += (winner == 0)
                    self.results[j, i] += (winner == 1)

    def get_results(self):
        for i in range(self.m):
            wins = np.sum(self.results[i, :] > self.results[:, i])
            losses = np.sum(self.results[i, :] < self.results[:, i])
            print(f"Agent {i}: {wins}-{losses}")