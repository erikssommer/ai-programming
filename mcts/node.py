import graphviz


class Node:
    def __init__(self, player, state, parent=None):
        self.player = player
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def get_reward(self):
        """
        Return the reward of the state represented by the node
        """
        # TODO: needs more work, temporary solution
        if self.player == 1:
            return 1
        else:
            return -1
        #return self.state.reward()

    def is_game_over(self):
        """
        Return True if the game represented by the state is over, False otherwise
        """
        return self.state.is_game_over()

    def apply_action(self, action):
        """
        Apply an action to the state represented by the node
        """
        # Get the next player
        next_player = self.get_next_player()

        # Create a new node representing the next state of the game
        next_node = Node(
            next_player, self.state.apply_action(action), parent=self)

        # Add the new node to the list of children of the current node
        self.children.append(next_node)

        return next_node

    def get_legal_moves(self):
        """
        Return the legal moves for the state represented by the node
        """
        return self.state.get_legal_actions()

    def get_next_player(self) -> None:
        return self.player % 2 + 1
    
    def visualize_tree(self, graph=None):
        """ 
        Visualize the tree structure of the MCTS tree (for debugging purposes)
        """
        if graph is None:
            graph = graphviz.Digraph()
        
        graph.node(str(id(self)), label=f'Player: {self.player}\nVisits: {self.visits}\nRewards: {self.rewards}\nState: {self.state}')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph
    
    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
