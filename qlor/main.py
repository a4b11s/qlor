import datetime
import logging
import multiprocessing
import os
import gymnasium
import torch

from vizdoom import gymnasium_wrapper  # This import will register all the environments

from qlor.epsilon import Epsilon
from qlor.trainer import Trainer
from qlor.utils import print_into_middle_of_terminal


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
        video_folder=f"videos/{datetime.datetime.now().isoformat(" ", timespec='seconds')}",
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    screen_shape = envs.single_observation_space["screen"].shape
    screen_shape = (1, 120, 160)  # (screen_shape[2], screen_shape[0], screen_shape[1])
    print("Screen shape:", screen_shape)

    epsilon = Epsilon(
        start=1.0,
        end=0.01,
        decay=100_000,
    )

    trainer = Trainer(
        envs=envs,
        val_env=val_env,
        epsilon=epsilon,
        device=device,
    )

    try:
        trainer.checkpoint_manager.load()
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found. Starting from scratch.")

    
    print_into_middle_of_terminal("Training started")
    trainer.train()


if __name__ == "__main__":
    print(os.getpid())
    train()
