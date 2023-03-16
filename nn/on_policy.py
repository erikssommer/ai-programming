import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from utility.read_config import config

# Static values for activation functions
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "linear": nn.Identity(),
}

# Static values for optimizers
OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}


class OnPolicy(nn.Module):
    def __init__(self,
                 states=config.board_size ** 2,
                 actions=config.board_size ** 2,
                 hidden_layers=config.hidden_layers,
                 neurons_per_layer=config.neurons_per_layer,
                 lr=config.lr,
                 activation=config.activation,
                 optimizer=config.optimizer,
                 loss=nn.CrossEntropyLoss(),
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

        # Add output layer
        layers.append(nn.Linear(neurons_per_layer[-1], actions))
        layers.append(nn.Softmax(dim=0))  # Add Softmax layer

        self.nn = nn.Sequential(*layers)

        self.losses = []
        self.accuracy = []
        self.print = config.plot_accuracy

        self.optimizer = OPTIMIZERS.get(optimizer)(self.parameters(), lr=lr)

        self.loss = loss

        if load:
            self.load_state_dict(torch.load(model_path))
            self.eval()

    def forward(self, x):
        return self.nn(x)

    def train_step(self, batch):

        roots, distributions = zip(*batch)

        states = torch.tensor([root.state.get_state_flatten()
                              for root in roots], dtype=torch.float32)

        dicts = []
        for index, dist in enumerate(distributions):
            distribution = []
            root, act = dist

            copy_act = act.copy()

            valid = roots[index].state.get_validity_of_children()

            for index, bin in enumerate(valid):
                if bin == 1:
                    distribution.append(copy_act.pop(0))
                else:
                    distribution.append(0)

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

    def rollout_action(self, node):
        state = torch.tensor(node.state.get_state_flatten(), dtype=torch.float32)
        predictions = self(state)
        legal = torch.tensor(node.state.get_validity_of_children(), dtype=torch.float32)
        index = torch.argmax(torch.multiply(predictions, legal)).item()
        return node.state.get_children()[index]


    def best_action(self, game):
        value = torch.tensor(game.get_state_flatten(), dtype=torch.float32)
        argmax = torch.multiply(torch.softmax(self(value), dim=0), torch.tensor(
            game.get_validity_of_children())).argmax().item()
        action = game.get_children()[argmax]
        return action

    def save(self, path):
        torch.save(self.state_dict(), path)
