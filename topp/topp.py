# The Tournament of Progressive Policies (TOPP)
from agent import Agent
import numpy as np

class TOPP:
    def __init__(self, m, g):
        self.m = m
        self.g = g
        self.agents = []
        self.results = np.zeros((m, m))

    def add_agents(self, m):
        for i in range(m):
            pollicy_path = f"agent_{i}.pt"
            self.agents.append(Agent(i+1, pollicy_path))
    
    def run_turnament(self):
        for i in range(self.m):
            for j in range(i+1, self.m):
                # Play a series of G games between agents i and j
                for game in range(self.g):
                    # Initialize the game
                    game = Game(size=6)
                    current_player = 0

                    # Play the game until it is over
                    while not game.is_over():
                        # Get the move from the current player's agent
                        agent = self.agents[current_player]
                        state = game.get_state()
                        move = agent.make_move(state)

                        # Make the move on the board
                        game.make_move(move)
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