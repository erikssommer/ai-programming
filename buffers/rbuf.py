import random
from collections import deque
from typing import Deque
import numpy as np
from utils.matrix import transform
from utils.read_config import config

# Replay buffer for storing training cases for neural network

class RBUF:
    """
    Replay buffer for storing training cases for neural network
    """
    def __init__(self, max_size=256):
        self.buffer = deque([], maxlen=max_size)
        self.max_size = max_size

    def get(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer

        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=batch_size)

    def add_case(self, case):
        player, game_state, distribution = case

        state = transform(player, game_state)

        if player == 2:
            distribution = np.array(distribution).reshape(config.board_size, config.board_size).T.flatten().tolist()

        self.buffer.append((state, distribution))
    
    def clear(self):
        self.buffer = deque([], maxlen=self.max_size)
