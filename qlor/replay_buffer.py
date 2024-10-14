from collections import deque
import os
import queue
import numpy as np
import torch
import h5py as h5
import threading


class ReplayBuffer:
    def __init__(self, max_size, max_ram_size, path, device, batch_size, prefetch=10):
        self.max_size = max_size
        self.path = path
        self.batch_size = batch_size

        self.prefetch_batches = queue.Queue(maxsize=prefetch)
        self.device = device

        self.disk_pointer = 0

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.h5_file = h5.File(path, "w")
        self.h5_file.create_dataset(
            "experiences",
            shape=(max_size, 5),
            maxshape=(None, 5),
            dtype=h5.vlen_dtype(np.float32),
        )
        self.h5_file.close()
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def add(self, state, action, reward, next_state, done):
        state = self._add_prepare(state)
        action = self._add_prepare(action)
        reward = self._add_prepare(reward)
        next_state = self._add_prepare(next_state)
        done = self._add_prepare(done)

        self._save_batch_to_disk([state, action, reward, next_state, done])

    def sample(self):
        return self.prefetch_batches.get()

    def __len__(self):
        return self.h5_file["experiences"].shape[0]

    def _add_prepare(self, data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data

    def __getitem__(self, idx):
        return self.h5_file["experiences"][idx]

    @staticmethod
    def map_batch(batch, device):
        state_batch = torch.cat(
            [experience[0].unsqueeze(0) for experience in batch]
        ).to(device)
        action_batch = torch.tensor(
            [experience[1] for experience in batch], device=device
        ).long()
        reward_batch = torch.tensor(
            [experience[2] for experience in batch],
            device=device,
            dtype=torch.float32,
        )
        next_state_batch = torch.cat(
            [experience[3].unsqueeze(0) for experience in batch]
        ).to(device)
        done_batch = torch.tensor(
            [experience[4] for experience in batch],
            device=device,
            dtype=torch.float32,
        )

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def _prefetch(self):
        while True:
            if self.prefetch_batches.qsize() < self.prefetch_batches.maxsize:
                loaded_batch = self._load_batch_from_disk()
                sampled = []

                for exp in loaded_batch:
                    sampled.append(
                        [torch.tensor(field, device=self.device) for field in exp]
                    )

                self.prefetch_batches.put(sampled)

    def _save_batch_to_disk(self, batch):
        batch_size = len(batch)
        if self.disk_pointer + batch_size <= self.max_size:
            self._save_to_disk(
                slice=(self.disk_pointer, self.disk_pointer + batch_size), data=batch
            )
            self.disk_pointer += batch_size
        else:
            self.disk_pointer = 0
            self._save_to_disk(
                slice=(self.disk_pointer, self.disk_pointer + batch_size), data=batch
            )
            self.disk_pointer += batch_size

    def _save_to_disk(self, slice, data):
        with h5.File(self.path, "a") as h5_file:
            h5_file["experiences"][slice] = data

    def _load_batch_from_disk(self):
        random_idx = np.sort(
            np.random.choice(
                np.arange(self.max_size), size=self.batch_size, replace=False
            )
        )
        
        with h5.File(self.path, "r") as h5_file:
            if len(h5_file["experiences"]) < self.batch_size:
                return h5_file["experiences"][: len(h5_file["experiences"])]

            return h5_file["experiences"][random_idx]
