import random
from collections import deque
from typing import Deque
import numpy as np

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
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append(case)


    """def __init__(self, size):
        self.size = size
        self.buffer = []
    
    def add_case(self, training_case):
        #print(training_case)
        if len(self.buffer) < self.size:
            self.buffer.append(training_case)
        else:
            self.buffer.pop(0)
            self.buffer.append(training_case)
    
    def get(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer

        #weights = [1 / i for i in range(1, len(self.buffer) + 1)]
        #return random.choices(self.buffer, weights=weights, k=batch_size)

        return random.sample(self.buffer, batch_size)"""
    
    def clear(self):
        self.buffer = []