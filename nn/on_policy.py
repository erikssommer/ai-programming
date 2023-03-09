import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from IPython import display
from IPython.utils import io
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
    def __init__(self, states, actions, hidden_size, lr, activation, optimizer, loss=nn.CrossEntropyLoss()):
        plt.ion()
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(states, hidden_size),
            ACTIVATIONS.get(activation),
            nn.Linear(hidden_size, hidden_size*2),
            ACTIVATIONS.get(activation),
            nn.Linear(hidden_size*2, hidden_size),
            ACTIVATIONS.get(activation),
            nn.Linear(hidden_size, actions),
            nn.Softmax(dim=0)
        )
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation

        layers = []
        
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation())
        
        # Add hidden layers
        for i in range(num_layers-1):  # Subtract 1 for input layer
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, num_classes))
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)
        """

        self.losses = []
        self.accuracy = []
        self.print = config.plot_accuracy
        
        self.optimizer = OPTIMIZERS.get(optimizer)(self.parameters(), lr=lr)
        
        self.loss = loss

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
        accuracy = (preds.argmax(dim=1) == targets.argmax(dim=1)).float().mean()
        self.accuracy.append(accuracy.item())

        loss.backward()
        self.optimizer.step()

        if self.print:
            with io.capture_output():
                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.clf()
                plt.title('Training...')
                plt.xlabel('Number of Games')
                plt.ylabel('Accuracy')
                plt.plot(self.accuracy, label='Accuracy')
                plt.show(block=False)
                plt.pause(.1)
