import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from utility.read_config import config
from managers.state_manager import StateManager
from utils.matrix import transform

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
                 states,
                 actions,
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
        self.actions = actions
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation

        layers = []

        """self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        layers.append(nn.Linear(states//2, neurons_per_layer[0]))
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

        self.nn = nn.Sequential(*layers)"""

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(actions, actions),
            nn.Softmax(dim=1)
        )

        self.losses = []
        self.accuracy = []
        self.print = config.plot_accuracy

        self.optimizer = OPTIMIZERS.get(optimizer)(self.parameters(), lr=lr)

        self.loss = LOSS.get(loss)

        if load:
            self.load_state_dict(torch.load(model_path))
            self.eval()

    def forward(self, x):
        x = self.conv(x)
        return x

    def train_step(self, batch):

        states, distributions = zip(*batch)

        targets = torch.tensor(distributions, dtype=torch.float32)
        states = torch.tensor(states, dtype=torch.float32)

        self.train()
        preds = self(states)
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

    # TODO: Are these two functions the same?
    def rollout_action(self, state: StateManager):

        state_matrix = transform(state.get_player(), state.get_game_state())
        state_matrix = torch.tensor(state_matrix, dtype=torch.float32)

        predictions = self(state_matrix).squeeze()

        legal = state.get_validity_of_children()

        #index = predictions.argmax().item()
        if state.get_player() == 1:

            for i in range(len(legal)):
                if legal[i] == 0:
                    predictions[i] = -1

            #index = random.choices(indices, weights=predictions)[0]
            index = predictions.argmax().item()

        else:
            predictions = predictions.unflatten(0, (config.board_size, config.board_size))
            predictions = predictions.T
            predictions = predictions.flatten()

            for i in range(len(legal)):
                if legal[i] == 0:
                    predictions[i] = -1

            #index = random.choices(indices, weights=predictions)[0]
            index = predictions.argmax().item()

        return state.get_children()[index]

    def debug(self, state: StateManager):
        state_matrix = transform(state.get_player(), state.get_game_state())
        state_matrix = torch.tensor(state_matrix, dtype=torch.float32)

        print("state_matrix", state_matrix)

        predictions = self(state_matrix).squeeze()

        legal = state.get_validity_of_children()

        indices = range(len(predictions))

        # index = predictions.argmax().item()
        if state.get_player() == 1:
            for i in range(len(legal)):
                if legal[i] == 0:
                    predictions[i] = -1

            print("predictions", predictions)

            #index = random.choices(indices, weights=predictions)[0]

            index = predictions.argmax().item()

        else:
            predictions = predictions.unflatten(0, (config.board_size, config.board_size))
            predictions = predictions.T
            predictions = predictions.flatten()

            for i in range(len(legal)):
                if legal[i] == 0:
                    predictions[i] = -1

            print("predictions", predictions)

            #index = random.choices(indices, weights=predictions)[0]

            index = predictions.argmax().item()
        print("index", index)

        return state.get_children()[index]

    def best_action(self, state: StateManager):
        #input("press enter")
        state_matrix = transform(state.get_player(), state.get_game_state())
        state_matrix = torch.tensor(state_matrix, dtype=torch.float32)
        #print("player", state.get_player())
        #print("state_matrix", state_matrix)

        predictions = self(state_matrix).squeeze()

        legal = state.get_validity_of_children()

        # index = predictions.argmax().item()
        if state.get_player() == 1:

            for i, value in enumerate(legal):
                if value == 0:
                    predictions[i] = -1

            # index = random.choices(indices, weights=predictions)[0]
            index = predictions.argmax().item()

        else:
            predictions = predictions.unflatten(0, (config.board_size, config.board_size))
            predictions = predictions.T
            predictions = predictions.flatten()

            for i in range(len(legal)):
                if legal[i] == 0:
                    predictions[i] = -1

            # index = random.choices(indices, weights=predictions)[0]
            index = predictions.argmax().item()

        #print("predictions", predictions)
        #input("press enter")

        return state.get_children()[index]

    def get_action(self, state: list[int]):
        value = torch.tensor(state, dtype=torch.float32)
        pred = self(value)

        for index, element in enumerate(state[:1]):
            if element != 0:
                pred[index] = -1

        argmax = pred.argmax().item()

        row = argmax // config.oht_board_size

        col = argmax % config.oht_board_size

        return row, col

    def save(self, path):
        torch.save(self.state_dict(), path)
