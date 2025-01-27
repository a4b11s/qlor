import datetime
import math
import random
import numpy as np
import torch

from torchvision import transforms

from qlor.models.agent import Agent
from qlor.models.autoencoder import Autoencoder
from qlor.modules.tensor_stack import TensorStack
from qlor.trainer.base_trainer import BaseTrainer
from qlor.modules.hyperparameters import HyperParameters
from qlor.replay_buffer.replay_buffer import ReplayBuffer


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
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((120, 160)),
            ]
        )

        self.replay_buffer = ReplayBuffer(
            max_size=self.hyperparameters.experience_replay_maxlen,
            batch_size=self.hyperparameters.batch_size,
            prefetch=24,
            replay_buffer_path=replay_buffer_path,
            num_envs=envs.num_envs,
            device=device,
        )

        self.autoencoder = Autoencoder(
            (self.hyperparameters.screen_stack_depth, 120, 160),
            self.hyperparameters.hidden_dim,
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

        self.screen_stack = TensorStack(
            stack_depth=self.hyperparameters.screen_stack_depth,
            batch_size=envs.num_envs,
            screen_shape=(1, 120, 160),
        )
        self.val_screen_stack = TensorStack(
            stack_depth=self.hyperparameters.screen_stack_depth,
            batch_size=1,
            screen_shape=(1, 120, 160),
        )

    def autoencoder_train_step(self, batch_size=None):
        loss = []

        for _ in range(self.hyperparameters.autoencoder_extra_steps):
            batch = self.replay_buffer.sample_batch(batch_size)
            state_batch = batch["observation"]
            step_loss = self.autoencoder.train_on_batch(
                state_batch=state_batch,
                optimizer=self.autoencoder_optimizer,
                loss_fn=self.autoencoder_loss,
            )
            loss.append(step_loss)

        return np.mean(loss)

    def agent_train_step(self, batch_size=None):
        batch = self.replay_buffer.sample_batch(batch_size)

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
        done_flags = torch.zeros(len(observation), device=self.device, dtype=torch.bool)
        self.screen_stack.clear()
        self.screen_stack.add_batch(observation, done_flags)

        while self.step < max_steps:
            current_state = self.screen_stack.get()
            encoded_state = self.autoencoder.encoder(current_state)
            policy = self.agent(encoded_state)
            actions = self.epsilon_greedy_action(policy, self.epsilon())
            next_observations, rewards, done_flags = self.execute_actions(actions)
            self.screen_stack.add_batch(next_observations, done_flags)
            self.replay_buffer.store_experience(
                current_state=current_state,
                next_observations=self.screen_stack.get(),
                actions=actions,
                rewards=rewards,
                done_flags=done_flags,
            )

            if self._should_train_agent():
                loss = self.agent_train_step()

            if self._should_train_autoencoder():
                autoencoder_loss = self.autoencoder_train_step()

            if done_flags.any():
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

        dones_flags = terminateds | truncateds
        dones_flags = torch.tensor(
            dones_flags, device=self.device, dtype=torch.bool
        ).squeeze()

        return next_observations, rewards, dones_flags

    def validate(self, max_steps=1000):
        observation, _ = self.val_env.reset()
        self.val_screen_stack.clear()
        observation = self.augment_observation(observation)
        done_flags = torch.zeros(len(observation), device=self.device, dtype=torch.bool)
        self.val_screen_stack.add_batch(observation, done_flags)

        rewards = []

        while len(rewards) < max_steps:
            current_state = self.val_screen_stack.get()
            hidden_state = self.autoencoder.encoder(current_state)
            policy = self.agent(hidden_state)
            actions = policy.argmax(dim=1).item()

            observation, reward, terminated, truncated, _ = self.val_env.step(actions)
            observation = self.augment_observation(observation)
            self.val_screen_stack.add_batch(observation, done_flags)

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
        return len(self.replay_buffer) > self.hyperparameters.batch_size * 2

    def _should_update_target_agent(self):
        return (
            self.step % self.hyperparameters.target_update_frequency == 0
            and self.step > 0
        )

    def _update_target_agent_if_needed(self):
        if self._should_update_target_agent():
            self.target_agent.load_state_dict(self.agent.state_dict())
