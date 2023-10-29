from typing import Callable
from functools import partial

import numpy as np

class ReplayBuffer:
    capacity: int
    size: int
    index: int
    data: dict[str, np.ndarray]

    def __init__(
        self,
        capacity: int,
        **data: np.ndarray,
    ):
        self.data = {
            key: np.zeros((capacity, *shape_dtype.shape), dtype=shape_dtype.dtype)
            for key, shape_dtype in data.items()
        }
        self.capacity = capacity
        self.size = 0
        self.index = 0

    def push(self, **kwargs: np.ndarray):
        for key, value in kwargs.items():
            self.data[key][self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch(self, batch_size: int, rng: np.ndarray) -> dict[str, np.ndarray] | None:
        if self.size < batch_size:
            return None

        indices = np.random.randint(0, self.size, batch_size)

        return {
            key: self.data[key][indices]
            for key in self.data
        }
