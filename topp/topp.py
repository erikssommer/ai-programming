from topp.agent import Agent
from game.nim import NimGame
import numpy as np
import os
import random

# The Tournament of Progressive Policies (TOPP)


class TOPP:
    def __init__(self, m, g):
        self.m = m+1  # Number of saved anets
        self.g = g  # Number of games to play between each pair of agents
        self.agents: Agent = []

    def add_agents(self):
        policy_path = f"./nn_models/"
        for file in os.listdir(policy_path):
            if file.endswith(".pt"):
                self.agents.append(Agent(policy_path, file))

    def run_turnament(self):
        for i in range(self.m):
            for j in range(i+1, self.m):
                starting_player = random.choice([i, j])
                # Play a series of G games between agents i and j
                for game in range(self.g):
                    # Initialize the game
                    game = NimGame(NimGame.generate_state(4))

                    current_player = starting_player

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

                    if winner == starting_player:
                        print("Starting player won")

                    if winner == i:
                        self.agents[i].add_win()
                        self.agents[j].add_loss()
                    elif winner == j:
                        self.agents[j].add_win()
                        self.agents[i].add_loss()

                    # Swap the starting player
                    if starting_player == i:
                        starting_player = j
                    else:
                        starting_player = i

    def get_results(self):
        for i in range(self.m):
            agent = self.agents[i]
            print(f"{agent.player}: {agent.win}-{agent.loss}")
