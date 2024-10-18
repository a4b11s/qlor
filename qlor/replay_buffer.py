from collections import deque
import os
import queue
import numpy as np
import torch
import h5py as h5
import threading
import logging


class ReplayBuffer:
    def __init__(self, max_size, h5_path, image_shape, device, batch_size, prefetch=50):
        self.max_size = max_size
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.prefetch_batches = queue.Queue(maxsize=prefetch)
        self.save_queue = queue.Queue(maxsize=prefetch)
        self.device = device

        self.length = 0

        self.disk_pointer = 0

        self._init_h5_file()
        self.lock = threading.Lock()
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.save_thread = threading.Thread(target=self._prefetch_save)
        self._run_threads()

    def add(self, state, action, reward, next_state, done):
        state = self._add_prepare(state)
        action = self._add_prepare(action)
        reward = self._add_prepare(reward)
        next_state = self._add_prepare(next_state)
        done = self._add_prepare(done)

        logging.debug(
            f"Adding experience to buffer: {state.shape}, {action}, {reward}, {next_state.shape}, {done}"
        )

        experience = np.array([[state, action, reward, next_state, done]], dtype=object)
        self._save_batch_to_disk(experience)

    def sample(self):
        print(self.prefetch_batches.qsize())
        return self.prefetch_batches.get()

    def __len__(self):
        return self.length

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
            if not self.prefetch_batches.full() and self.length >= self.batch_size:
                loaded_batch = self._load_batch_from_disk()
                sampled = []

                for exp in loaded_batch:
                    sampled.append(
                        [torch.tensor(field, device=self.device) for field in exp]
                    )

                self.prefetch_batches.put(sampled)

    def _prefetch_save(self):
        while True:
            if not self.save_queue.empty():
                slice, data = self.save_queue.get()
                self._save_to_disk(slice, data)

    def _save_batch_to_disk(self, batch):
        batch_size = len(batch)
        if self.disk_pointer + batch_size <= self.max_size:
            slice = (self.disk_pointer, self.disk_pointer + batch_size)
            self.disk_pointer += batch_size
        else:
            self.disk_pointer = 0
            slice = (self.disk_pointer, self.disk_pointer + batch_size)
            self.disk_pointer += batch_size

        self.save_queue.put((slice, batch))
        self.length += batch_size

    def _save_to_disk(self, slice, data):
        self.lock.acquire()
        with h5.File(self.h5_path, "a") as h5_file:
            h5_file["states"][slice[0] : slice[1]] = np.array(
                [experience[0] for experience in data]
            )
            h5_file["actions"][slice[0] : slice[1]] = np.array(
                [experience[1] for experience in data]
            )
            h5_file["rewards"][slice[0] : slice[1]] = np.array(
                [experience[2] for experience in data]
            )
            h5_file["next_states"][slice[0] : slice[1]] = np.array(
                [experience[3] for experience in data]
            )
            h5_file["dones"][slice[0] : slice[1]] = np.array(
                [experience[4] for experience in data]
            )
        self.lock.release()

    def _load_batch_from_disk(self):
        if self.length < self.batch_size:
            raise ValueError("Not enough samples in the buffer")

        indx = np.sort(np.random.choice(self.length, self.batch_size, replace=False))

        with h5.File(self.h5_path, "r") as h5_file:
            states = np.empty(
                (self.batch_size, *h5_file["states"].shape[1:]),
                dtype=h5_file["states"].dtype,
            )
            actions = np.empty((self.batch_size,), dtype=h5_file["actions"].dtype)
            rewards = np.empty((self.batch_size,), dtype=h5_file["rewards"].dtype)
            next_states = np.empty(
                (self.batch_size, *h5_file["next_states"].shape[1:]),
                dtype=h5_file["next_states"].dtype,
            )
            dones = np.empty((self.batch_size,), dtype=h5_file["dones"].dtype)

            # Fill the pre-allocated arrays
            for i, index in enumerate(indx):
                states[i] = h5_file["states"][index]
                actions[i] = h5_file["actions"][index]
                rewards[i] = h5_file["rewards"][index]
                next_states[i] = h5_file["next_states"][index]
                dones[i] = h5_file["dones"][index]

        return list(zip(states, actions, rewards, next_states, dones))

    def _init_h5_file(self):
        # Ensure the directory exists
        if not os.path.exists(self.h5_path):
            os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)
        # If the file exists but is not a valid HDF5 file, remove it
        if os.path.exists(self.h5_path) and not h5.is_hdf5(self.h5_path):
            os.remove(self.h5_path)

        state_shape = (self.max_size,) + self.image_shape
        action_shape = (self.max_size,)
        reward_shape = (self.max_size,)
        done_shape = (self.max_size,)

        with h5.File(self.h5_path, "w") as h5_file:
            h5_file.create_dataset(
                "states",
                shape=state_shape,
                maxshape=state_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "next_states",
                shape=state_shape,
                maxshape=state_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "actions",
                shape=action_shape,
                maxshape=action_shape,
                chunks=True,
                dtype=np.int64,  # Assuming integer actions
            )
            h5_file.create_dataset(
                "rewards",
                shape=action_shape,
                maxshape=reward_shape,
                chunks=True,
                dtype=np.float32,
            )
            h5_file.create_dataset(
                "dones",
                shape=action_shape,
                maxshape=done_shape,
                chunks=True,
                dtype=np.bool,
            )

    def _run_threads(self):
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        self.save_thread.daemon = True
        self.save_thread.start()
