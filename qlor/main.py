import gymnasium
import gymnasium.wrappers.record_video
import numpy as np
from vizdoom import gymnasium_wrapper  # This import will register all the environments
from qlor.agent import Agent
from torch import nn, optim
import torch

from qlor.epsilon import Epsilon
from qlor.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

envs = gymnasium.make_vec(
    "VizdoomCorridor-v0",  # or any other environment id
    num_envs=16,
    # render_mode="human",
)  # or any other environment id

screen_shape = envs.single_observation_space["screen"].shape
screen_shape = (screen_shape[2], screen_shape[0], screen_shape[1])
action_dim = envs.single_action_space.n

agent = Agent(screen_shape, action_dim)
target_agent = Agent(screen_shape, action_dim).to(device)
target_agent.load_state_dict(agent.state_dict())

optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Define the loss criterion


def train():
    epsilon = Epsilon(
        start=1.0,
        end=0.01,
        decay=0.999,
    )

    trainer = Trainer(
        agent=agent,
        target_agent=target_agent,
        envs=envs,
        optimizer=optimizer,
        epsilon=epsilon,
        criterion=criterion,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":

    print(f"Using device: {torch.get_default_device()}")
    train()
