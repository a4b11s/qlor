import logging
import multiprocessing
import os
import time
import gymnasium
import numpy as np
from vizdoom import gymnasium_wrapper  # This import will register all the environments
from qlor.agent import Agent
from torch import nn, optim
import torch

from qlor.epsilon import Epsilon
from qlor.trainer import Trainer


def train():
    multiprocessing.set_start_method("spawn", force=True)

    is_fork = multiprocessing.get_start_method() == "fork"
    print("Forking is", "enabled" if is_fork else "disabled")
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    torch.set_default_device(device)

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.ERROR)

    print(f"Using device: {torch.get_default_device()}")

    env_id = "VizdoomCorridor-v0"

    envs = gymnasium.make_vec(
        env_id,  # or any other environment id
        num_envs=32,
        # render_mode="human",
    )

    val_env = gymnasium.wrappers.RecordVideo(
        gymnasium.make(env_id, render_mode="rgb_array"),
        video_folder=f"videos/{time.time()}",
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    screen_shape = envs.single_observation_space["screen"].shape
    screen_shape = (1, 120, 160)  # (screen_shape[2], screen_shape[0], screen_shape[1])
    action_dim = envs.single_action_space.n
    print("Screen shape:", screen_shape)
    agent = Agent(screen_shape, action_dim)
    target_agent = Agent(screen_shape, action_dim).to(device)
    target_agent.load_state_dict(agent.state_dict())

    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epsilon = Epsilon(
        start=1.0,
        end=0.01,
        decay=5000,
    )

    trainer = Trainer(
        agent=agent,
        target_agent=target_agent,
        envs=envs,
        val_env=val_env,
        optimizer=optimizer,
        epsilon=epsilon,
        criterion=criterion,
        device=device,
    )

    try:
        trainer.load("checkpoint")
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found. Starting from scratch.")

    trainer.train()


if __name__ == "__main__":
    print(os.getpid())
    train()
