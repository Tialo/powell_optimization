import numpy as np
from abc import ABCMeta, abstractmethod


class Problem(metaclass=ABCMeta):
    def __init__(self, size: int, l: np.ndarray = None, r: np.ndarray = None):
        self.size = size
        if l is None:
            l = np.full(size, -100)
        if r is None:
            r = np.full(size, 100)
        self.l = l
        self.r = r

    @abstractmethod
    def f(self, p: np.ndarray) -> float:
        pass


class EasomProblem(Problem):
    def __init__(self):
        size = 2
        super().__init__(size)

    def f(self, p):
        x, y = p
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


class CrossProblem(Problem):
    def __init__(self):
        size = 2
        l = np.full(size, -10)
        r = np.full(size, 10)
        super().__init__(size)

    def f(self, p):
        x, y = p
        return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2)) / np.pi))) + 1) ** 0.1


class SchafferProblem(Problem):
    def __init__(self):
        size = 2
        super().__init__(size)

    def f(self, p):
        x, y = p
        return 0.5 + (np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2


class EggHolderProblem(Problem):
    def __init__(self):
        size = 2
        l = np.full(size, -512.0)
        r = np.full(size, 512.0)
        super().__init__(size, l, r)

    def f(self, p):
        x, y = p
        return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


class HolderTableProblem(Problem):
    def __init__(self):
        size = 2
        l = np.full(size, -10.0)
        r = np.full(size, 10.0)
        super().__init__(size, l, r)

    def f(self, p):
        x, y = p
        return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)))


class RosenbrockProblem(Problem):
    def __init__(self):
        size = 2
        super().__init__(size)

    def f(self, p: np.ndarray) -> float:
        x, y = p
        return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


if __name__ == '__main__':
    pass
