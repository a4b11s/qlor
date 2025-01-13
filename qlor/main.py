import datetime
import logging
import multiprocessing
import os
import gymnasium
import torch

from vizdoom import gymnasium_wrapper  # This import will register all the environments

from qlor.modules.epsilon import Epsilon
from qlor.modules.hyperparameters import HyperParameters
from qlor.trainer.trainer import Trainer
from qlor.utils import print_into_middle_of_terminal


def initialize_process_configuration():
    multiprocessing.set_start_method("spawn", force=True)

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    torch.set_default_device(device)
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.ERROR)

    return device


def print_configuration():
    print_into_middle_of_terminal("Configuration")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.get_default_device()}")
    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
    print(f"Process ID: {os.getpid()}")
    print("*" * os.get_terminal_size().columns)


def train():
    timestamp = datetime.datetime.now()
    video_folder = f"videos/{timestamp.isoformat(' ', timespec='seconds')}"
    episode_trigger = lambda _: True
    env_id = "VizdoomCorridor-v0"
    device = initialize_process_configuration()
    print_configuration()

    envs = gymnasium.make_vec(
        env_id,
        num_envs=16,
    )

    val_env = gymnasium.wrappers.RecordVideo(
        gymnasium.make(env_id, render_mode="rgb_array"),
        video_folder=video_folder,
        episode_trigger=episode_trigger,
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
    
    hp = HyperParameters(
        experience_replay_maxlen=800_000,
    )

    trainer = Trainer(
        envs=envs,
        val_env=val_env,
        epsilon=epsilon,
        device=device,
        hyperparameters=hp,
    )

    try:
        trainer.checkpoint_manager.load()
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found. Starting from scratch.")

    print_into_middle_of_terminal("Training started")
    trainer.train()


if __name__ == "__main__":
    train()
