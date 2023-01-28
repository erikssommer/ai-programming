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
    
    def reward(self):
        """
        Return the reward of the state represented by the node
        """
        return self.state.reward()

    def is_game_over(self):
        """
        Return True if the game represented by the state is over, False otherwise
        """
        return self.state.is_game_over()
    
    def apply_action(self, action):
        """
        Apply an action to the state represented by the node
        """
        self.state.apply_action(action)