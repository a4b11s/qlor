from torch import nn


class Agent(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Agent, self).__init__()
        self.flatten = nn.Flatten()
        self.actor = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.actor(x)
        return nn.functional.softmax(x, dim=-1)
    
