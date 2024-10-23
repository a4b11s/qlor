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

    def on_step(self, step):
        if step % self.save_interval == 0 and step > 0:
            self.save(step)

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

    def save(self, step):
        trainer_config_path = self._save_trainer_config(step)
        metrics_path = self._save_metrics(step)
        experience_replay_path = self._save_experience_replay(step)
        networks_path = self._save_networks(step)

        manifest = {
            "trainer_config": trainer_config_path,
            "metrics": metrics_path,
            "experience_replay": experience_replay_path,
            "networks": networks_path,
            "step": step,
        }

        with open(os.path.join(self.checkpoint_dir, f"manifest_{step}.json"), "w") as f:
            json.dump(manifest, f)

        if self.max_to_keep is not None:
            while self._count_checkpoints() > self.max_to_keep:
                self._delete_oldest_checkpoint()

        return manifest

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

        os.remove(manifest_path)
        os.remove(manifest["trainer_config"])
        os.remove(manifest["metrics"])
        os.remove(manifest["experience_replay"])
        for network in manifest["networks"].values():
            os.remove(network)
