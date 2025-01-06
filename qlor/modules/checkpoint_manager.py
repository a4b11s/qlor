import os
import json
import pickle

import torch


class CheckpointManager:
    def __init__(self, trainer, checkpoint_dir, max_to_keep=None, save_interval=1000):
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.save_interval = save_interval

        os.makedirs(checkpoint_dir, exist_ok=True)

        self._keep_max_if_needed()

    def on_step(self, step):
        if step % self.save_interval == 0 and step > 0:
            self.save(step)

    def save(self, step):
        hyperparameters_path = self._save_hyperparameters(step)
        trainer_config_path = self._save_trainer_config(step)
        metrics_path = self._save_metrics(step)
        # TODO: Implement experience replay saving. Now it's stoping PC
        # experience_replay_path = self._save_experience_replay(step)
        networks_path = self._save_networks(step)

        manifest = {
            "hyperparameters": hyperparameters_path,
            "trainer_config": trainer_config_path,
            "metrics": metrics_path,
            # "experience_replay": experience_replay_path,
            "networks": networks_path,
            "step": step,
        }

        with open(os.path.join(self.checkpoint_dir, f"manifest_{step}.json"), "w") as f:
            json.dump(manifest, f)

        self._keep_max_if_needed()

        return manifest

    def load(self, step=None):
        if step is not None:
            return self._load_checkpoint_by_step(step)
        else:
            return self._load_latest_checkpoint()

    def _load_hyperparameters(self, path):
        self.trainer.hyperparameters.set_config(json.load(open(path, "r")))
        return self.trainer.hyperparameters

    def _load_trainer_config(self, path):
        self.trainer.set_config(pickle.load(open(path, "rb")))
        return self.trainer

    def _load_metrics(self, path):
        with open(path, "rb") as f:
            self.trainer.metrics_manager.set_config(pickle.load(f))

    def _load_experience_replay(self, path):
        self.trainer.experience_replay.loads(path)
        return self.trainer.experience_replay

    def _load_networks(self, path):
        self.trainer.agent.load_state_dict(torch.load(path["agent"], weights_only=True))
        self.trainer.autoencoder.load_state_dict(
            torch.load(path["autoencoder"], weights_only=True)
        )
        return self.trainer.agent, self.trainer.autoencoder

    def _load_checkpoint_by_step(self, step):
        manifest_path = os.path.join(self.checkpoint_dir, f"manifest_{step}.json")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self._load_hyperparameters(manifest["hyperparameters"])
        self._load_trainer_config(manifest["trainer_config"])
        self._load_metrics(manifest["metrics"])
        # self._load_experience_replay(manifest["experience_replay"])
        self._load_networks(manifest["networks"])

    def _load_latest_checkpoint(self):
        checkpoints = [
            name
            for name in os.listdir(self.checkpoint_dir)
            if name.startswith("manifest_")
        ]

        if len(checkpoints) < 1:
            raise FileNotFoundError("No checkpoints found.")

        latest_checkpoint = max(
            checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        step = int(latest_checkpoint.split("_")[1].split(".")[0])

        return self._load_checkpoint_by_step(step)

    def _save_trainer_config(self, step):
        trainer_config_path = os.path.join(
            self.checkpoint_dir, f"trainer_config_{step}.pkl"
        )

        with open(trainer_config_path, "wb") as f:
            pickle.dump(self.trainer.get_config(), f)

        return trainer_config_path

    def _save_metrics(self, step):
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_{step}.pkl")

        metrics_config = self.trainer.metrics_manager.get_config()

        with open(metrics_path, "wb") as f:
            pickle.dump(metrics_config, f)

        return metrics_path

    def _save_experience_replay(self, step):
        experience_replay_path = os.path.join(
            self.checkpoint_dir, f"experience_replay_{step}/"
        )

        self.trainer.experience_replay.dumps(experience_replay_path)

        return experience_replay_path

    def _save_networks(self, step):
        agent = self.trainer.agent
        autoencoder = self.trainer.autoencoder

        agent_path = os.path.join(self.checkpoint_dir, f"agent_{step}.pth")
        autoencoder_path = os.path.join(self.checkpoint_dir, f"autoencoder_{step}.pth")

        torch.save(agent.state_dict(), agent_path)
        torch.save(autoencoder.state_dict(), autoencoder_path)

        return {
            "agent": agent_path,
            "autoencoder": autoencoder_path,
        }

    def _save_hyperparameters(self, step):
        hyperparameters_path = os.path.join(
            self.checkpoint_dir, f"hyperparameters_{step}.json"
        )
        hyperparameters = self.trainer.hyperparameters.get_config()

        with open(hyperparameters_path, "w") as f:
            json.dump(hyperparameters, f)

        return hyperparameters_path

    def _count_checkpoints(self):
        return len(
            [
                name
                for name in os.listdir(self.checkpoint_dir)
                if name.startswith("manifest_")
            ]
        )

    def _delete_oldest_checkpoint(self):
        checkpoints = [
            name
            for name in os.listdir(self.checkpoint_dir)
            if name.startswith("manifest_")
        ]
        oldest_checkpoint = min(
            checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        manifest_path = os.path.join(self.checkpoint_dir, oldest_checkpoint)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for path in manifest.values():
            if isinstance(path, int):
                continue
            elif isinstance(path, dict):
                for subpath in path.values():
                    if os.path.exists(subpath):
                        os.remove(subpath)
            elif os.path.exists(path):
                os.remove(path)

        os.remove(manifest_path)

    def _keep_max_if_needed(self):
        if self.max_to_keep is not None:
            while self._count_checkpoints() > self.max_to_keep:
                self._delete_oldest_checkpoint()
