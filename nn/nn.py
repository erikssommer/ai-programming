import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from IPython import display
from IPython.utils import io


class Actor(nn.Module):
    def __init__(self, states, actions, hidden_size, optimizer=optim.Adam, loss=nn.CrossEntropyLoss(), lr=0.001):
        plt.ion()
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions),
            nn.Softmax(dim=0)
        )

        self.losses = []
        self.accuracy = []
        self.print = True

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, x):
        return self.nn(x)

    def train_step(self, batch):
        roots, distributions = zip(*batch)

        states = torch.tensor([root.state.get_state_flatten() for root in roots], dtype=torch.float32)

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

            #print(stat.get_validity_of_children())
            #print(act)
            dicts.append(distribution)

        targets = torch.tensor(dicts, dtype=torch.float32)

        self.train()

        preds = self(states)

        """
        for i in range(len(batch)):
            actions = roots[i].state.get_children()

            index = actions.index(best_moves[i].action)

            targets[i][index] = roots[i].rewards
            targets[i] = targets[i]"""

        #targets = torch.softmax(targets, dim=1)

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
                plt.plot(self.accuracy)
                plt.show(block=False)
                plt.pause(.1)




