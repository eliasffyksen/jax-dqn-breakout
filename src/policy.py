import jax
import jax.numpy as jnp
import flax.linen as nn

class NatureCNN(nn.Module):
    features: int = 512

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)

        return x
    
