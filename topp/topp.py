from topp.agent import Agent
from nim.nim import NimGame
import numpy as np
import os
import random

# The Tournament of Progressive Policies (TOPP)
class TOPP:
    def __init__(self, m, g):
        self.m = m+1 # Number of saved anets
        self.g = g # Number of games to play between each pair of agents
        self.agents: Agent = []
        self.results = np.zeros((m+1, m+1))

    def add_agents(self):
        policy_path = f"./nn_models/"
        for file in os.listdir(policy_path):
            if file.endswith(".pt"):
                self.agents.append(Agent(policy_path, file))
    
    def run_turnament(self):
        for i in range(self.m):
            for j in range(i+1, self.m):
                # Play a series of G games between agents i and j
                for game in range(self.g):
                    # Initialize the game
                    game = NimGame(NimGame.generate_state(4))
                    current_player = random.choice([i, j])
                    # Play the game until it is over
                    while not game.is_game_over():
                        # Get the move from the current player's agent
                        agent = self.agents[current_player]
                        move = agent.make_move(game)
                        
                        # Make the move on the board
                        game.apply_action_self(move)

                        if not game.is_game_over():
                            if current_player == i:
                                current_player = j
                            else:
                                current_player = i

                    # Record the result of the game
                    # winner = game.get_winner()

                    winner = current_player

                    self.results[i, j] += (winner == i)
                    self.results[j, i] += (winner == j)
                    

    def get_results(self):
        for i in range(self.m):
            wins = np.sum(self.results[i, :] > self.results[:, i])
            losses = np.sum(self.results[i, :] < self.results[:, i])
            print(f"{self.agents[i]}: {wins}-{losses}")