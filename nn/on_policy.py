import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from utility.read_config import config
from mcts.mcts import Node
from game.game import Game

# Static values for activation functions
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "linear": nn.Identity(),
    "leaky_relu": nn.LeakyReLU(.2),
}

# Static values for optimizers
OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

LOSS = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss()
}


class OnPolicy(nn.Module):
    def __init__(self,
                 states=config.board_size ** 2 + 1,
                 actions=config.board_size ** 2,
                 hidden_layers=config.hidden_layers,
                 neurons_per_layer=config.neurons_per_layer,
                 lr=config.lr,
                 activation=config.activation,
                 optimizer=config.optimizer,
                 loss=config.loss,
                 load=False,
                 model_path=None):
        super().__init__()
        """
        self.nn = nn.Sequential(
            nn.Linear(states, neurons_per_layer),
            ACTIVATIONS.get(activation),
            nn.Linear(neurons_per_layer, neurons_per_layer*2),
            ACTIVATIONS.get(activation),
            nn.Linear(neurons_per_layer*2, neurons_per_layer),
            ACTIVATIONS.get(activation),
            nn.Linear(neurons_per_layer, actions),
            nn.Softmax(dim=0)
        )
        """

        self.states = states
        self.neurons_per_layer = neurons_per_layer
        self.actions = actions
        self.activation = activation

        layers = []

        # Add input layer
        layers.append(nn.Linear(states, neurons_per_layer[0]))
        layers.append(ACTIVATIONS.get(activation))

        # Add hidden layers
        for i in range(hidden_layers-1):
            layers.append(
                nn.Linear(neurons_per_layer[i], neurons_per_layer[i+1]))
            layers.append(ACTIVATIONS.get(activation))
            layers.append(nn.Dropout(p=0.2))

        # Add output layer
        layers.append(nn.Linear(neurons_per_layer[-1], actions))
        layers.append(nn.Softmax(dim=0))  # Add Softmax layer

        self.nn = nn.Sequential(*layers)

        self.losses = []
        self.accuracy = []
        self.print = config.plot_accuracy

        self.optimizer = OPTIMIZERS.get(optimizer)(self.parameters(), lr=lr)

        self.loss = LOSS.get(loss)

        if load:
            self.load_state_dict(torch.load(model_path))
            self.eval()

    def forward(self, x):
        return self.nn(x)

    def train_step(self, batch):

        roots, distributions = zip(*batch)

        states = []

        for root in roots:
            state = [root.state.player] + root.state.get_state_flatten()
            states.append(state)

        states = torch.tensor(states, dtype=torch.float32)

        dicts = []
        for index, dist_tuple in enumerate(distributions):
            distribution = []
            root, dist = dist_tuple

            copy_dist = dist.copy()

            valid = roots[index].state.get_validity_of_children()

            for _, validity in enumerate(valid):
                if validity == 1:
                    distribution.append(copy_dist.pop(0))
                else:
                    distribution.append(-1)

            dicts.append(distribution)

        targets = torch.tensor(dicts, dtype=torch.float32).softmax(dim=1)

        self.train()
        preds = self(states)

        self.optimizer.zero_grad()
        loss = self.loss(preds, targets)
        self.losses.append(loss.item())
        accuracy = (preds.argmax(dim=1) ==
                    targets.argmax(dim=1)).float().mean()
        self.accuracy.append(accuracy.item())

        loss.backward()
        self.optimizer.step()

        if self.print:
            plt.clf()
            plt.ion()
            plt.title('Training...')
            plt.xlabel('Number of Games')
            plt.ylabel('Accuracy')
            plt.plot(self.accuracy, label='Accuracy')
            plt.show(block=False)
            plt.pause(.1)

    def rollout_action(self, node: Node):
        state = torch.tensor([node.state.player] + node.state.get_state_flatten(), dtype=torch.float32)
        predictions = self(state)
        legal = torch.tensor(node.state.get_validity_of_children(), dtype=torch.float32)

        for index, value in enumerate(legal):
            if value == 0:
                predictions[index] = -1000

        index = torch.argmax(predictions).item()
        return node.state.get_children()[index]

    def best_action(self, game: Game):
        state = torch.tensor([game.player] + game.get_state_flatten(), dtype=torch.float32)

        pred = self(state)

        for index, value in enumerate(game.get_validity_of_children()):
            if value == 0:
                pred[index] = -1000

        argmax = pred.argmax().item()
        action = game.get_children()[argmax]
        return action

    def save(self, path):
        torch.save(self.state_dict(), path)
