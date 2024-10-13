import datetime
import json
import os
import random
import timeit
import numpy as np
import torch
import pickle

from qlor.epsilon import Epsilon
from qlor.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
        self, agent, target_agent, envs, optimizer, epsilon, criterion, device
    ):
        self.agent: torch.nn.Module = agent
        self.target_agent: torch.nn.Module = target_agent

        self.envs = envs
        self.action_dim = envs.single_action_space.n

        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.epsilon: Epsilon = epsilon
        self.gamma = 0.99

        self.batch_size = 256

        self.target_update_frequency = 200
        self.validation_frequency = 1000

        self.print_frequency = 10
        self.save_frequency = 5000
        self.experience_replay_maxlen = 2000

        self.episode = 0
        self.step = 0

        self.start_time = datetime.datetime.now()

        self.experience_replay = ReplayBuffer(
            self.experience_replay_maxlen, self.device
        )

        self.save_path = "checkpoint"

        self.metrics = {}

        self.config_field = [
            "optimizer",
            "criterion",
            "device",
            "epsilon",
        ]

    def train_batch(self, batch_size):
        batch = self.experience_replay.sample(batch_size)

        state_batch = torch.cat(
            [experience[0].unsqueeze(0) for experience in batch]
        ).to(self.device)
        action_batch = torch.tensor(
            [experience[1] for experience in batch], device=self.device
        ).long()
        reward_batch = torch.tensor(
            [experience[2] for experience in batch],
            device=self.device,
            dtype=torch.float32,
        )
        next_state_batch = torch.cat(
            [experience[3].unsqueeze(0) for experience in batch]
        ).to(self.device)

        # Current Q values
        policy_batch = self.agent(state_batch)
        current_q_values = policy_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_policy_batch = self.agent(next_state_batch)
            next_actions = next_policy_batch.argmax(dim=1)
            target_policy_batch = self.target_agent(next_state_batch)
            next_q_values = target_policy_batch.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, max_steps=1_000_000):
        self.start_time = datetime.datetime.now()
        observation, _ = self.envs.reset()
        loss = 0

        while self.step < max_steps:
            current_state = self.augment_observation(observation)
            policy = self.agent(current_state)
            actions = self.epsilon_greedy_action(policy, self.epsilon())
            next_observations, rewards, terminateds, truncateds, _ = self.envs.step(
                actions
            )

            for i in range(self.envs.num_envs):
                self.experience_replay.add(
                    current_state[i],
                    actions[i],
                    rewards[i],
                    self.augment_observation(next_observations)[i],
                )

            if len(self.experience_replay) > self.batch_size:
                loss = self.train_batch(self.batch_size)

            observation = next_observations

            self.update_metrics("loss", loss, mode="replace")
            self.update_metrics("reward", np.mean(rewards), mode="replace")
            self.update_metrics("avg_loss", loss, mode="average")
            self.update_metrics("avg_reward", np.mean(rewards), mode="average")

            if np.any(terminateds) or np.any(truncateds):
                self.episode += 1

            self.on_step_end()

    def on_step_end(self):
        self.step += 1
        self.epsilon.update_epsilon(self.step)

        self.update_metrics("epsilon", self.epsilon(), mode="replace")
        self.update_metrics("step", self.step, mode="replace")
        
        elapsed_time = datetime.datetime.now() - self.start_time
        self.update_metrics("elapsed_time", str(elapsed_time), mode="replace")

        torch.cuda.empty_cache()
        if self.step % self.print_frequency == 0 and self.step > 0:
            self.print_metrics()

        if self.step % self.target_update_frequency == 0 and self.step > 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

        if self.step % self.save_frequency == 0 and self.step > 0:
            self.save(self.save_path)

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
        screen = screen.permute(0, 3, 1, 2)

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
