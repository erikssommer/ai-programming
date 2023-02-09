class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def add_child(self, node):
        node.parent = self
        self.children.append(node)
    
    def update(self, reward):
        self.visits += 1
        self.rewards += reward
    
    def get_reward(self):
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
        next_state = self.state.take_action(action)
        
        # Create a new node representing the next state of the game
        next_node = Node(next_state, parent=self)
        
        # Add the new node to the list of children of the current node
        self.children.append(next_node)
        
        return next_node

    def get_legal_moves(self):
        """
        Return the legal moves for the state represented by the node
        """
        return self.state.get_legal_moves()
