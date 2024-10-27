class HyperParameters:
    def __init__(
        self,
        hidden_dim: int = 256,
        batch_size: int = 128,
        gamma: float = 0.99,
        autoencoder_extra_steps: int = 10,
        autoencoder_update_frequency: int = 100,
        target_update_frequency: int = 1000,
        experience_replay_maxlen: int = 2_000_000,
    ) -> None:
        """
        Initializes the hyperparameters.

        :param hidden_dim: The number of hidden units in the Q-network and autoencoder.
        :param batch_size: The number of samples to draw from the replay buffer.
        :param gamma: The discount factor.
        :param autoencoder_extra_steps: The number of extra steps to train the autoencoder on.
        :param autoencoder_update_frequency: The frequency of updating the autoencoder.
        :param target_update_frequency: The frequency of updating the target network.
        :param experience_replay_maxlen: The maximum length of the experience replay buffer.
        """

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.autoencoder_extra_steps = autoencoder_extra_steps
        self.autoencoder_update_frequency = autoencoder_update_frequency
        self.target_update_frequency = target_update_frequency
        self.experience_replay_maxlen = experience_replay_maxlen

    def get_config(self) -> dict:
        """
        Returns the configuration of the hyperparameters.

        :return: The configuration of the hyperparameters.
        """

        return {
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "autoencoder_extra_steps": self.autoencoder_extra_steps,
            "autoencoder_update_frequency": self.autoencoder_update_frequency,
            "target_update_frequency": self.target_update_frequency,
            "experience_replay_maxlen": self.experience_replay_maxlen,
        }

    def set_config(self, config: dict) -> None:
        """
        Sets the configuration of the hyperparameters.

        :param config: The configuration of the hyperparameters.
        """

        self.hidden_dim = config["hidden_dim"]
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.autoencoder_extra_steps = config["autoencoder_extra_steps"]
        self.autoencoder_update_frequency = config["autoencoder_update_frequency"]
        self.target_update_frequency = config["target_update_frequency"]
        self.experience_replay_maxlen = config["experience_replay_maxlen"]

    def __str__(self) -> str:
        return f"HyperParameters(hidden_dim={self.hidden_dim}, batch_size={self.batch_size}, gamma={self.gamma}, autoencoder_extra_steps={self.autoencoder_extra_steps}, autoencoder_update_frequency={self.autoencoder_update_frequency}, target_update_frequency={self.target_update_frequency}, experience_replay_maxlen={self.experience_replay_maxlen})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperParameters):
            return False
        return (
            self.hidden_dim == other.hidden_dim
            and self.batch_size == other.batch_size
            and self.gamma == other.gamma
            and self.autoencoder_extra_steps == other.autoencoder_extra_steps
            and self.autoencoder_update_frequency == other.autoencoder_update_frequency
            and self.target_update_frequency == other.target_update_frequency
            and self.experience_replay_maxlen == other.experience_replay_maxlen
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(
            (
                self.hidden_dim,
                self.batch_size,
                self.gamma,
                self.autoencoder_extra_steps,
                self.autoencoder_update_frequency,
                self.target_update_frequency,
                self.experience_replay_maxlen,
            )
        )

    def __copy__(self) -> object:
        return HyperParameters(
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            gamma=self.gamma,
            autoencoder_extra_steps=self.autoencoder_extra_steps,
            autoencoder_update_frequency=self.autoencoder_update_frequency,
            target_update_frequency=self.target_update_frequency,
            experience_replay_maxlen=self.experience_replay_maxlen,
        )

    def __deepcopy__(self, memo: dict) -> object:
        return HyperParameters(
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            gamma=self.gamma,
            autoencoder_extra_steps=self.autoencoder_extra_steps,
            autoencoder_update_frequency=self.autoencoder_update_frequency,
            target_update_frequency=self.target_update_frequency,
            experience_replay_maxlen=self.experience_replay_maxlen,
        )
