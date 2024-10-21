import datetime
import json
import os
import random
import tempfile
import time
import numpy as np
import torch
import pickle

from torchvision import transforms
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement
from tensordict.tensordict import TensorDict

from qlor.agent import Agent
from qlor.autoencoder import Autoencoder
from qlor.env_model import EnvModel
from qlor.epsilon import Epsilon


class Trainer(object):
    episode = 0
    step = 0
    start_time = None
    save_path = "checkpoint"
    metrics = {}

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
        self.save_frequency = 5000

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
        self.env_model = EnvModel(self.hidden_dim, self.action_dim).to(device)
        self.agent = Agent(self.hidden_dim, self.action_dim).to(device)
        self.target_agent = Agent(self.hidden_dim, self.action_dim).to(device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.autoencoder_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=1e-3
        )
        self.env_model_optimizer = torch.optim.Adam(
            self.env_model.parameters(), lr=1e-3
        )
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

        self.criterion = torch.nn.MSELoss()
        self.autoencoder_loss = torch.nn.MSELoss()
        self.env_model_loss = torch.nn.MSELoss()

    def train_agent_on_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        state_batch = batch["observation"]
        action_batch = batch["action"]
        reward_batch = batch["rewards"]
        next_state_batch = batch["next_observation"]
        done_batch = batch["done"]

        self.agent.train()

        # Current Q values
        policy_batch = self.agent(state_batch)

        current_q_values = policy_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        target_q_values = self.calculate_target_q_values(
            next_state_batch, reward_batch, done_batch
        )

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.agent.eval()

        return loss.item()

    def train_autoencoder_on_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        state_batch = batch["observation"]

        self.autoencoder.train()
        decoded = self.autoencoder(state_batch)

        loss = self.autoencoder_loss(decoded, state_batch)

        self.autoencoder_optimizer.zero_grad()
        loss.backward()
        self.autoencoder_optimizer.step()
        self.autoencoder.eval()

        return loss.item()

    def train_env_model_on_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        state_batch = batch["observation"]
        action_batch = batch["action"]
        next_state_batch = batch["next_observation"]

        self.env_model.train()
        input = torch.cat([state_batch, action_batch], dim=1)
        predicted_next_state = self.env_model(input)

        loss = self.env_model_loss(predicted_next_state, next_state_batch)

        self.env_model_optimizer.zero_grad()
        loss.backward()
        self.env_model_optimizer.step()
        self.env_model.eval()

        return loss.item()

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

            # if len(self.experience_replay) > self.batch_size * 2:
            # loss = self.train_agent_on_batch()
            # env_model_loss = self.train_env_model_on_batch()

            # if self.step % self.autoencoder_update_frequency == 0:
            #     autoencoder_loss = self.train_autoencoder_on_batch()

            observation = next_observations

            if np.any(terminateds) or np.any(truncateds):
                self.episode += 1

            self.on_step_end(
                {
                    "loss": loss,
                    "reward": np.mean(rewards),
                    "env_model_loss": 0, #env_model_loss,
                    "autoencoder_loss": 0, #autoencoder_loss,
                }
            )

    def on_step_end(self, logs):
        torch.cuda.empty_cache()

        self.step += 1
        elapsed_time = datetime.datetime.now() - self.start_time
        self.epsilon.update_epsilon(self.step)

        self.update_metrics("epsilon", self.epsilon(), mode="replace")
        self.update_metrics("step", self.step, mode="replace")
        self.update_metrics("buffer_size", len(self.experience_replay), mode="replace")
        self.update_metrics("loss", logs["loss"], mode="replace")
        self.update_metrics("reward", logs["reward"], mode="replace")
        self.update_metrics("env_model_loss", logs["env_model_loss"], mode="replace")
        self.update_metrics(
            "autoencoder_loss", logs["autoencoder_loss"], mode="replace"
        )
        self.update_metrics("elapsed_time", str(elapsed_time), mode="replace")

        if self.step % self.validation_frequency == 0:
            val = self.validate()
            self.update_metrics("val_reward", val, mode="replace")

        if self.step % self.print_frequency == 0 and self.step > 0:
            self.print_metrics()

        if self.step % self.target_update_frequency == 0 and self.step > 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

        if self.step % self.save_frequency == 0 and self.step > 0:
            self.save(self.save_path)

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

    def mpc_planning(self, state, horizon=10):
        raise NotImplementedError

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

        for metric_name, metric_value in self.metrics.items():
            if metric_name[0] == "_":
                continue

            if isinstance(metric_value, float):
                print_string += f", {metric_name}: {metric_value:.3f}"
            else:
                print_string += f", {metric_name}: {metric_value}"

        print(print_string)

    def update_metrics(self, metrics_name, value, mode="average"):
        if metrics_name not in self.metrics:
            self.metrics[metrics_name] = 0

        if mode == "sum":
            self.metrics[metrics_name] += value

        elif mode == "average":
            # old_value = self.metrics[metrics_name]
            # self.metrics[metrics_name] = (old_value + value) / 2
            if "_" + metrics_name not in self.metrics:
                self.metrics["_" + metrics_name] = []

            self.metrics["_" + metrics_name].append(value)
            self.metrics[metrics_name] = np.mean(self.metrics["_" + metrics_name])

        elif mode == "replace":
            self.metrics[metrics_name] = value

    def save_config(self, path):
        with open(path, "wb") as f:
            pickle.dump({field: getattr(self, field) for field in self.config_field}, f)

    def load_config(self, path):
        with open(path, "rb") as f:
            config = pickle.load(f)
            for field, value in config.items():
                setattr(self, field, value)

    def save_metrics(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.metrics, f)

    def load_metrics(self, path):
        with open(path, "rb") as f:
            self.metrics = pickle.load(f)

    def save_experience_replay(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.experience_replay, f)

    def load_experience_replay(self, path):
        with open(path, "rb") as f:
            self.experience_replay = pickle.load(f)

    def save_agent(self, path):
        torch.save(self.agent.state_dict(), path)

    def load_agent(self, path):
        dict = torch.load(path)
        self.agent.load_state_dict(dict)
        self.target_agent.load_state_dict(dict)

    def save(self, path):
        return  # Disable saving for now
        print(f"Saving checkpoint to {path}")

        if not os.path.exists(path):
            os.makedirs(path)

        manifest = {
            "config_path": path + "/config.pkl",
            "metrics_path": path + "/metrics.pkl",
            "experience_replay_path": path + "/experience_replay.pkl",
            "agent_path": path + "/agent.pth",
            "datetime": str(datetime.datetime.now()),
        }

        self.save_config(manifest["config_path"])
        self.save_metrics(manifest["metrics_path"])
        # self.save_experience_replay(manifest["experience_replay_path"])
        self.save_agent(manifest["agent_path"])

        with open(path + "/manifest.json", "w") as f:
            f.write(json.dumps(manifest, indent=4))

        print("Checkpoint saved")

    def load(self, path):
        with open(path + "/manifest.json", "r") as f:
            manifest = json.loads(f.read())

        self.load_config(manifest["config_path"])
        self.load_metrics(manifest["metrics_path"])
        # self.load_experience_replay(manifest["experience_replay_path"])
        self.load_agent(manifest["agent_path"])

    @staticmethod
    def map_batch(batch, device):
        state_batch = torch.tensor(batch["state"], dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch["action"], device=device).long()
        reward_batch = torch.tensor(batch["reward"], device=device, dtype=torch.float32)
        next_state_batch = torch.tensor(
            batch["next_state"], dtype=torch.float32, device=device
        )
        done_batch = torch.tensor(batch["done"], device=device, dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
