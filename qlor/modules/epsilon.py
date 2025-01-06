import math


class Epsilon:
    def __init__(self, start, end, decay):
        self.epsilon_start = start
        self.epsilon_end = end
        self.epsilon_decay = decay
        self.epsilon = start

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)

    def __call__(self) -> float:
        return self.epsilon

    def __repr__(self) -> str:
        return f"Epsilon(epsilon={self.epsilon})"

    def __str__(self) -> str:
        return f"Epsilon(epsilon={self.epsilon})"

    def __eq__(self, o: object) -> bool:
        if type(o) == float:
            return self.epsilon == o
        if isinstance(o, Epsilon):
            return self.epsilon == o.epsilon
        return False

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __lt__(self, o: object) -> bool:
        if type(o) == float:
            return self.epsilon < o
        if isinstance(o, Epsilon):
            return self.epsilon < o.epsilon
        return False

    def __le__(self, o: object) -> bool:
        if type(o) == float:
            return self.epsilon <= o
        if isinstance(o, Epsilon):
            return self.epsilon <= o.epsilon
        return False

    def __gt__(self, o: object) -> bool:
        if type(o) == float:
            return self.epsilon > o
        if isinstance(o, Epsilon):
            return self.epsilon > o.epsilon
        return False

    def __ge__(self, o: object) -> bool:
        if type(o) == float:
            return self.epsilon >= o
        if isinstance(o, Epsilon):
            return self.epsilon >= o.epsilon
        return False
