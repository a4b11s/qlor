import datetime
import random
import numpy as np
import torch

from torchvision import transforms
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement
from tensordict.tensordict import TensorDict

from qlor.agent import Agent
from qlor.autoencoder import Autoencoder
from qlor.checkpoint_manager import CheckpointManager
from qlor.epsilon import Epsilon
from qlor.metric import Metric


class Trainer(object):
    episode = 0
    step = 0
    start_time = None
    save_path = "checkpoint"
    metrics = {
        "epsilon": Metric("epsilon", "set"),
        "step": Metric("step", "set"),
        "buffer_size": Metric("buffer_size", "set"),
        "loss": Metric("loss", "average"),
        "reward": Metric("reward", "average"),
        "autoencoder_loss": Metric("autoencoder_loss", "average"),
        "elapsed_time": Metric("elapsed_time", "set"),
        "val_reward": Metric("val_reward", "set"),
    }

    def __init__(
        self,
        envs,
        val_env,
        epsilon,
        replay_buffer_path,
        device,
    ):
        self.envs = envs
        self.val_env = val_env
        self.action_dim = envs.single_action_space.n
        self.device = device
        self.epsilon: Epsilon = epsilon
        self.checkpoint_manager = CheckpointManager(
            self, self.save_path, max_to_keep=5, save_interval=1000
        )

        # Hyperparameters
        self.gamma = 0.99
        self.hidden_dim = 256
        self.batch_size = 128
        self.experience_replay_maxlen = 2_000_000
        self.target_update_frequency = 1000
        self.autoencoder_update_frequency = 100

        # Training parameters
        self.validation_frequency = 5000
        self.print_frequency = 100

        self.experience_replay = ReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=self.experience_replay_maxlen,
                scratch_dir=replay_buffer_path,
            ),
            batch_size=self.batch_size,
            prefetch=4,
            sampler=SamplerWithoutReplacement(),
        )

        self.config_field = [
            "optimizer",
            "criterion",
            "epsilon",
            "step",
            "episode",
            "start_time",
        ]

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((120, 160)),
            ]
        )

        self.autoencoder = Autoencoder((1, 120, 160), self.hidden_dim).to(device)
        self.agent = Agent(self.hidden_dim, self.action_dim).to(device)
        self.target_agent = Agent(self.hidden_dim, self.action_dim).to(device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.autoencoder_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=1e-3
        )
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        self.autoencoder_loss = torch.nn.MSELoss()
        self.env_model_loss = torch.nn.MSELoss()

    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        return batch

    def train_agent_on_batch(self, batch_size=None):
        batch = self.sample_batch(batch_size)

        state_batch = batch["observation"]
        action_batch = batch["action"]
        reward_batch = batch["rewards"]
        next_state_batch = batch["next_observation"]
        done_batch = batch["done"]

        state_batch = self.autoencoder.encoder(state_batch)
        next_state_batch = self.autoencoder.encoder(next_state_batch)

        return self.agent.train_on_batch(
            state_batch=state_batch,
            action_batch=action_batch,
            reward_batch=reward_batch,
            next_state_batch=next_state_batch,
            done_batch=done_batch,
            criterion=self.criterion,
            optimizer=self.optimizer,
            calculate_target_q_values=self.calculate_target_q_values,
        )

    def train(self, max_steps=1_000_000):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()

        observation, _ = self.envs.reset()
        loss = 0

        observation = self.augment_observation(observation)  # bschw

        while self.step < max_steps:
            current_state = observation
            policy = torch.zeros(32, self.action_dim)  # self.agent(current_state)
            actions = self.epsilon_greedy_action(policy, self.epsilon())
            next_observations, rewards, terminateds, truncateds, _ = self.envs.step(
                actions
            )

            next_observations = self.augment_observation(next_observations)

            data_dict = TensorDict(
                {
                    "observation": current_state,
                    "next_observation": next_observations,
                    "action": torch.tensor(actions),
                    "rewards": torch.tensor(rewards, dtype=torch.float32),
                    "done": torch.tensor(terminateds) | torch.tensor(truncateds),
                },
                device=self.device,
                batch_size=self.envs.num_envs,
            )

            self.experience_replay.extend(data_dict)

            if len(self.experience_replay) > self.batch_size * 2:
                loss = self.train_agent_on_batch()

            if self.step % self.autoencoder_update_frequency == 0:
                batch = self.sample_batch()
                autoencoder_loss = self.autoencoder.train_on_batch(
                    state_batch=batch["observation"],
                    optimizer=self.autoencoder_optimizer,
                    loss_fn=self.autoencoder_loss,
                )

            observation = next_observations

            if np.any(terminateds) or np.any(truncateds):
                self.episode += 1

            self.on_step_end(
                {
                    "loss": loss,
                    "reward": np.mean(rewards),
                    "env_model_loss": 0,  # env_model_loss,
                    "autoencoder_loss": autoencoder_loss,
                }
            )

    def on_step_end(self, logs):
        torch.cuda.empty_cache()

        self.step += 1
        elapsed_time = datetime.datetime.now() - self.start_time
        self.epsilon.update_epsilon(self.step)

        self.metrics["epsilon"].update(self.epsilon())
        self.metrics["step"].update(self.step)
        self.metrics["buffer_size"].update(len(self.experience_replay))
        self.metrics["loss"].update(logs["loss"])
        self.metrics["reward"].update(logs["reward"])
        self.metrics["autoencoder_loss"].update(logs["autoencoder_loss"])
        self.metrics["elapsed_time"].update(str(elapsed_time))

        self.checkpoint_manager.on_step(self.step)

        if self.step % self.validation_frequency == 0:
            val = self.validate()
            self.metrics["val_reward"].update(val)

        if self.step % self.print_frequency == 0 and self.step > 0:
            self.print_metrics()

        if self.step % self.target_update_frequency == 0 and self.step > 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

    def validate(self, max_steps=1000):
        observation, _ = self.val_env.reset()
        rewards = []

        while len(rewards) < max_steps:
            current_state = self.augment_observation(observation)
            policy = self.agent(current_state)
            actions = policy.argmax(dim=1).item()

            observation, reward, terminated, truncated, _ = self.val_env.step(actions)

            rewards.append(reward)

            if terminated or truncated:
                break

        return np.mean(rewards)

    def calculate_target_q_values(self, next_state_batch, reward_batch, done_batch):
        with torch.no_grad():
            next_policy_batch = self.agent(next_state_batch)
            next_actions = next_policy_batch.argmax(dim=1)
            target_policy_batch = self.target_agent(next_state_batch)
            next_q_values = target_policy_batch.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values * (
                1 - done_batch.float()
            )

        return target_q_values

    def epsilon_greedy_action(self, policy, epsilon):
        actions = []
        for p in policy:
            actions.append(
                random.randrange(self.action_dim)
                if random.random() < epsilon
                else p.argmax().item()
            )
        return np.array(actions)

    def augment_observation(self, observation):
        screen = observation["screen"]
        screen = torch.tensor(screen, device=self.device)  # bs w h c
        # From bshwc to bschw
        if screen.dim() < 4:
            screen = screen.unsqueeze(0)
        screen = screen.permute(0, 3, 1, 2)

        screen = self.transform(screen)

        return screen / 255.0

    def print_metrics(self):
        print_string = f"Episode: {self.episode}"

        for metric in self.metrics.values():
            print_string += str(metric) + " "

        print(print_string)

    def get_config(self):
        config = {field: getattr(self, field) for field in self.config_field}

        return config

    def set_config(self, config):
        for field, value in config.items():
            setattr(self, field, value)
