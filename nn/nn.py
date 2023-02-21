import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from IPython import display
from IPython.utils import io


class Actor(nn.Module):
    def __init__(self, states, actions, hidden_size, optimizer=optim.SGD, loss=nn.MSELoss(), lr=0.01):
        plt.ion()
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions),
            #nn.Softmax(dim=0)
        )

        self.losses = []

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, x):
        return self.nn(x)

    def train_step(self, batch):
        self.train()
        roots, distributions, best_moves = zip(*batch)

        states = torch.tensor([root.state.get_state_flatten() for root in roots], dtype=torch.float32)

        preds = self(states)
        targets = torch.clone(preds)

        for i in range(len(batch)):
            actions = roots[i].state.get_children()

            index = actions.index(best_moves[i].action)

            targets[i][index] = roots[i].rewards
            targets[i] = torch.multiply(torch.softmax(targets[i], dim=0),
                                        torch.tensor(roots[i].state.get_validity_of_children()))

        self.optimizer.zero_grad()

        loss = self.loss(preds, targets)
        loss.backward()

        self.losses.append(loss.item())
        self.optimizer.step()

        with io.capture_output() as captured:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.title('Training...')
            plt.xlabel('Number of Games')
            plt.ylabel('Loss')
            plt.plot(self.losses)
            plt.show(block=False)
            plt.pause(.1)




