"""
Replay Buffer

"""

import jax.numpy as jnp
import jax.random as jrandom

class ReplayBuffer():
    """A simple FIFO experience replay buffer for off-policy agents."""

    def __init__(self, buffer_size, state_dim, action_dim, key):
        """Initialize replay buffer with zeros."""
        self.rng = key  # rundom number generator
        data_point_dim = 2 * state_dim + action_dim + 1
        self.data_points = jnp.zeros((buffer_size, data_point_dim))
        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size
        return

    def store(self, data_tuple):
        """Store new experience."""
        data_point = jnp.hstack(data_tuple)
        self.data_points = self.data_points.at[self.ptr].set(data_point)
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)
        return

    def sample_batch(self, batch_size):
        """Sample past experience."""
        batch_size = min(batch_size, self.size)
        self.rng, rng_input = jrandom.split(self.rng)
        indexes = jrandom.randint(rng_input, shape=(batch_size,),
                                 minval=0, maxval=self.size)
        return self.data_points[indexes]


