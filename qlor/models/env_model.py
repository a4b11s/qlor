from torch import nn


class EnvModel(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(EnvModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x):
        return self.fc(x)
