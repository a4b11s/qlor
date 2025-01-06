import datetime
import os

from qlor.modules.checkpoint_manager import CheckpointManager
from qlor.modules.epsilon import Epsilon
from qlor.modules.hyperparameters import HyperParameters
from qlor.modules.metrics_manager import MetricsManager


class BaseTrainer:
    step = 0
    start_time = None
    save_path = "checkpoint"

    def __init__(
        self,
        envs,
        val_env,
        epsilon,
        device,
        hyperparameters: HyperParameters = None,
    ):
        self.envs = envs
        self.val_env = val_env
        self.action_dim = envs.single_action_space.n
        self.device = device
        self.epsilon: Epsilon = epsilon
        self.checkpoint_manager = CheckpointManager(
            self, self.save_path, max_to_keep=5, save_interval=50_000
        )

        self.hyperparameters = (
            hyperparameters
            if hyperparameters
            else HyperParameters(experience_replay_maxlen=1_000_000)
        )

        # Training parameters
        self.validation_frequency = 5000
        self.print_frequency = 100

        self.config_field = [
            "optimizer",
            "criterion",
            "epsilon",
            "step",
            "episode",
            "start_time",
        ]

        self.metrics_manager = MetricsManager(
            [
                {"name": "Episode", "mode": "set"},
                {"name": "epsilon", "mode": "set"},
                {"name": "step", "mode": "set"},
                {"name": "loss", "mode": "average"},
                {"name": "autoencoder_loss", "mode": "average"},
                {"name": "reward", "mode": "average"},
                {"name": "elapsed_time", "mode": "set"},
                {"name": "val_reward", "mode": "set"},
            ]
        )

    def _initialize_training(self):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()

    def _should_print_metrics(self):
        return self.step % self.print_frequency == 0 and self.step > 0

    def _should_validate(self):
        return self.step % self.validation_frequency == 0

    def _print_metrics_if_needed(self):
        if self._should_print_metrics():
            print(self.metrics_manager.get_string("\n"))
            print("*" * os.get_terminal_size().columns)

    def get_config(self):
        config = {field: getattr(self, field) for field in self.config_field}

        return config

    def set_config(self, config):
        for field, value in config.items():
            setattr(self, field, value)
