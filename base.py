from abc import ABCMeta, abstractmethod
import numpy as np


class Problem(metaclass=ABCMeta):
    def __init__(self, size: int, l: np.ndarray = None, r: np.ndarray = None):
        self.size = size
        if l is None:
            l = np.full(size, -100.0)
        if r is None:
            r = np.full(size, 100.0)
        self.l = l
        self.r = r

    @abstractmethod
    def f(self, p: np.ndarray) -> float:
        pass


class LineSearchMethod(metaclass=ABCMeta):
    @abstractmethod
    def search(self, problem: Problem, x0: np.ndarray, f0: float, d: np.ndarray) -> (float, float):
        pass


if __name__ == '__main__':
    pass
