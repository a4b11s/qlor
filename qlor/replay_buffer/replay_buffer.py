import torch
import torchrl

from tensordict.tensordict import TensorDict


class ReplayBuffer(object):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        prefetch: int,
        replay_buffer_path: str,
        num_envs: int,
        device,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.prefetch = prefetch
        self.replay_buffer_path = replay_buffer_path
        self.num_envs = num_envs
        self.device = device

        self.experience_replay: torchrl.data.ReplayBuffer = torchrl.data.ReplayBuffer(
            storage=torchrl.data.LazyMemmapStorage(
                max_size=self.max_size,
                scratch_dir=self.replay_buffer_path,
            ),
            batch_size=self.batch_size,
            prefetch=self.prefetch,
            sampler=torchrl.data.SamplerWithoutReplacement(),
        )

    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.experience_replay.sample(batch_size).to(self.device)

        return batch

    def store_experience(
        self, current_state, next_observations, actions, rewards, done_flags
    ):
        data_dict = self._create_data_dict(
            current_state, next_observations, actions, rewards, done_flags
        )
        self.experience_replay.extend(data_dict)

    def _create_data_dict(
        self,
        current_state,
        next_observations,
        actions,
        rewards,
        done_flags,
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
            batch_size=self.num_envs,
        )

    def __len__(self):
        return len(self.experience_replay)