from torch import nn

class Agent(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(Agent, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)
