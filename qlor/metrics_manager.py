from qlor.metric import Metric


class MetricsManager:
    metrics = {}

    def __init__(self, metrics_config: list) -> None:
        for metric_config in metrics_config:
            self.add_metric(metric_config["name"], metric_config["mode"])

    def add_metric(self, metric: Metric) -> None:
        self.metrics[metric.name] = metric

    def add_metric(self, name: str, mode: str) -> None:
        self.metrics[name] = Metric(name, mode)

    def update_metric(self, name: str, value: float) -> None:
        self.metrics[name].update(value)

    def update_many(self, data: dict) -> None:
        for name, value in data.items():
            self.update_metric(name, value)

    def get_string(self, separator=" ") -> str:
        string = ""

        for name, metric in self.metrics.items():
            string += f"{name}: {metric.value}{separator}"

        return string

    def __str__(self) -> str:
        return self.get_string(" | ")

    def get_config(self) -> dict:
        return {"metrics": [metric.get_config() for metric in self.metrics.values()]}

    def set_config(self, config: dict) -> None:
        for metric_config in config["metrics"]:
            self.add_metric(metric_config["name"], metric_config["mode"])
            self.metrics[metric_config["name"]].set_config(metric_config)
