from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, device):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

        self.device = device

    def add(self, state, action, reward, next_state):
        state = self._add_prepare(state)
        action = self._add_prepare(action)
        reward = self._add_prepare(reward)
        next_state = self._add_prepare(next_state)

        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size, device=None):
        _device = device

        if _device is None:
            _device = self.device

        if _device is None:
            _device = torch.device("cpu")

        idx = np.random.choice(
            np.arange(len(self.buffer)), size=batch_size, replace=False
        )

        sampled = []

        for i in idx:
            sampled.append(
                [torch.tensor(field, device=_device) for field in self.buffer[i]]
            )

        return sampled

    def __len__(self):
        return len(self.buffer)

    def _add_prepare(self, data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data

    def __getitem__(self, idx):
        return self.buffer[idx]
