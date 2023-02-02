from torch import nn
from torch import optim


class Actor(nn.Module):
    def __init__(self, states, actions, hidden_size, optimizer=optim.Adam, loss=nn.CrossEntropyLoss(), lr=0.001):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions),
            nn.Softmax(dim=1)
        )

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, x):
        return self.nn(x)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y, y_pred)
        loss.backward()
        self.optimizer.step()
