from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, donate_argnames=("buffer",))
def update_buffer(buffer: jax.Array, index: int, value: jax.Array) -> jax.Array:
    return buffer.at[index].set(value)
update_buffer: type(update_buffer)

class ReplayBuffer:
    capacity: int
    size: int
    index: int
    data: dict[str, jax.Array]

    def __init__(
        self,
        capacity: int,
        **data: dict[str, jax.ShapeDtypeStruct],
    ):
        self.data = {
            key: jnp.zeros((capacity, *shape_dtype.shape), dtype=shape_dtype.dtype)
            for key, shape_dtype in data.items()
        }
        self.capacity = capacity
        self.size = 0
        self.index = 0

    def push(self, **kwargs: dict[str, jax.Array]):
        for key, value in kwargs.items():
            self.data[key] = update_buffer(self.data[key], self.index, value)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch(self, batch_size: int, rng: jax.Array) -> dict[str, jax.Array] | None:
        if self.size < batch_size:
            return None

        indices = jax.random.randint(rng, (batch_size,), 0, self.size)

        return {
            key: self.data[key][indices]
            for key in self.data
        }
