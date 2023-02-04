import random

class RBUF:
    """
    Replay buffer for storing training cases for neural network
    """
    def __init__(self, size):
        self.size = size
        self.buffer = []
    
    def add(self, training_case):
        if len(self.buffer) < self.size:
            self.buffer.append(training_case)
        else:
            self.buffer.pop(0)
            self.buffer.append(training_case)
    
    def get(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer

        weights = [1 / i for i in range(1, len(self.buffer) + 1)]
        return random.choices(self.buffer, weights=weights, k=batch_size)