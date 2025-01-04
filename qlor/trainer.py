import datetime
import math
import random
import numpy as np
import torch

from torchvision import transforms
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement
from tensordict.tensordict import TensorDict

from qlor.agent import Agent
from qlor.autoencoder import Autoencoder
from qlor.base_trainer import BaseTrainer
from qlor.hyperparameters import HyperParameters


class Trainer(BaseTrainer):
    episode = 0

    def __init__(
        self,
        envs,
        val_env,
        epsilon,
        device,
        replay_buffer_path: str = "/tmp/qlor_rb/",
        hyperparameters: HyperParameters = None,
    ):
        super().__init__(
            envs=envs,
            val_env=val_env,
            epsilon=epsilon,
            device=device,
            hyperparameters=hyperparameters,
        )

        self.experience_replay = ReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=self.hyperparameters.experience_replay_maxlen,
                scratch_dir=replay_buffer_path,
            ),
            batch_size=self.hyperparameters.batch_size,
            prefetch=4,
            sampler=SamplerWithoutReplacement(),
        )

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((120, 160)),
            ]
        )

        self.autoencoder = Autoencoder(
            (1, 120, 160), self.hyperparameters.hidden_dim
        ).to(device)
        self.agent = Agent(self.hyperparameters.hidden_dim, self.action_dim).to(device)
        self.target_agent = Agent(self.hyperparameters.hidden_dim, self.action_dim).to(
            device
        )
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
            batch_size = self.hyperparameters.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        return batch

    def autoencoder_train_step(self, batch_size=None):
        loss = []

        for _ in range(self.hyperparameters.autoencoder_extra_steps):
            batch = self.sample_batch(batch_size)
            state_batch = batch["observation"]
            step_loss = self.autoencoder.train_on_batch(
                state_batch=state_batch,
                optimizer=self.autoencoder_optimizer,
                loss_fn=self.autoencoder_loss,
            )
            loss.append(step_loss)

        return np.mean(loss)

    def agent_train_step(self, batch_size=None):
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

    def train(self, max_steps=math.inf):
        self._initialize_training()

        observation, _ = self.envs.reset()
        loss = 0

        observation = self.augment_observation(observation)  # bschw

        while self.step < max_steps:
            current_state = observation
            encoded_state = self.autoencoder.encoder(current_state)
            policy = self.agent(encoded_state)
            actions = self.epsilon_greedy_action(policy, self.epsilon())
            next_observations, rewards, done_flags = self.execute_actions(actions)
            self.store_experience(
                current_state=current_state,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                done_flags=done_flags,
            )

            if self._should_train_agent():
                loss = self.agent_train_step()

            if self._should_train_autoencoder():
                autoencoder_loss = self.autoencoder_train_step()

            observation = next_observations

            if np.any(done_flags):
                self.episode += 1

            self.on_step_end(
                {
                    "loss": loss,
                    "reward": np.mean(rewards),
                    "autoencoder_loss": autoencoder_loss,
                }
            )

    def on_step_end(self, logs):
        torch.cuda.empty_cache()

        self.step += 1
        elapsed_time = datetime.datetime.now() - self.start_time
        self.epsilon.update_epsilon(self.step)

        self.metrics_manager.update_many(
            {
                "Episode": self.episode,
                "epsilon": self.epsilon(),
                "step": self.step,
                "elapsed_time": str(elapsed_time),
                **logs,
            }
        )

        self.checkpoint_manager.on_step(self.step)

        if self._should_validate():
            val = self.validate()
            self.metrics_manager.update_metric("val_reward", val)

        self._update_target_agent_if_needed()
        self._print_metrics_if_needed()

    def execute_actions(self, actions):
        next_observations, rewards, terminateds, truncateds, _ = self.envs.step(actions)
        next_observations = self.augment_observation(next_observations)

        return next_observations, rewards, terminateds | truncateds

    def store_experience(
        self, current_state, next_observations, actions, rewards, done_flags
    ):
        data_dict = self.create_data_dict(
            current_state, next_observations, actions, rewards, done_flags
        )
        self.experience_replay.extend(data_dict)

    def create_data_dict(
        self, current_state, next_observations, actions, rewards, done_flags
    ):
        return TensorDict(
            {
                "observation": current_state,
                "next_observation": next_observations,
                "action": torch.tensor(actions),
                "rewards": torch.tensor(rewards, dtype=torch.float32),
                "done": torch.tensor(done_flags),
            },
            device=self.device,
            batch_size=self.envs.num_envs,
        )

    def validate(self, max_steps=1000):
        observation, _ = self.val_env.reset()
        rewards = []

        while len(rewards) < max_steps:
            current_state = self.augment_observation(observation)
            hidden_state = self.autoencoder.encoder(current_state)
            policy = self.agent(hidden_state)
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
            target_q_values = (
                reward_batch
                + self.hyperparameters.gamma * next_q_values * (1 - done_batch.float())
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

    def _should_train_autoencoder(self):
        return self.step % self.hyperparameters.autoencoder_update_frequency == 0

    def _should_train_agent(self):
        return len(self.experience_replay) > self.hyperparameters.batch_size * 2

    def _should_update_target_agent(self):
        return (
            self.step % self.hyperparameters.target_update_frequency == 0
            and self.step > 0
        )

    def _update_target_agent_if_needed(self):
        if self._should_update_target_agent():
            self.target_agent.load_state_dict(self.agent.state_dict())
