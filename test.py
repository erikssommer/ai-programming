from mcts.node import Node

node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

node1.add_child(node2)
node1.add_child(node3)

print(len(node1.children) != 0)