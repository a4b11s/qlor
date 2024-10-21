import pickle


class Metric(object):
    modes = ["average", "sum", "set"]

    def __init__(self, name, mode):
        """
        Metric class to store the metric value and mode
        :param name: Name of the metric
        :param mode: Mode of the metric. Can be one of "average", "sum", "set"
        :return: Metric object
        """
        self.name = name
        if mode not in self.modes:
            raise ValueError(f"Mode must be one of {self.modes}")

        self.mode = mode

        self.value = 0
        self.count = None

        if mode == "average":
            self.count = 0

    def update(self, value):
        """
        Update the metric value based on the mode
        :param value: Value to update the metric with
        :return: None
        """
        if self.mode == "average":
            self.value = (self.value * self.count + value) / (self.count + 1)
            self.count += 1
        elif self.mode == "sum":
            self.value += value
        elif self.mode == "set":
            self.value = value

    def save(self, path):
        """
        Save the metric object to a file
        :param path: Path to save the metric object
        :return: None
        """
        with open(path, "wb") as f:
            pickle.dump(self.get_config(), f)

    def load(self, path):
        """
        Load the metric object from a file
        :param path: Path to load the metric object
        :return: None
        """
        with open(path, "rb") as f:
            config = pickle.load(f)
            self.set_config(config)

    def get_config(self):
        """
        Get the configuration of the metric object
        :return: Configuration of the metric object
        """
        return {
            "name": self.name,
            "mode": self.mode,
            "value": self.value,
            "count": self.count,
        }

    def set_config(self, config):
        """
        Set the configuration of the metric object
        :param config: Configuration to set the metric object
        :return: None
        """
        self.name = config["name"]
        self.mode = config["mode"]
        self.value = config["value"]
        self.count = config["count"]

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __repr__(self):
        return f"{self.name}: {self.value}"
