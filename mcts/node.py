class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def add_child(self, action, state):
        child = Node(self, state)
        self.children[action] = child
        return child
    
    def update(self, reward):
        self.visits += 1
        self.rewards += reward
    