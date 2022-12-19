"""
Replay Buffer
"""

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom


class ReplayBuffer():
    """
    Replay buffer for off-policy learing

    Alteration on publicly available module. Original module can be found on: https://github.com/chisarie/jax-agents/blob/master/jax_agents/common/data_processor.py
    """

    def __init__(self, buffer_size, state_dim, action_dim, key):
        """
        Initialize replay buffer
        :param buffer_size: buffer size [int]
        :param state_dim: state dimension [int]
        :param action_dim: action dimension [int]
        :param key: PRNGKey
        """
        self.rng = key  # rundom number generator
        data_point_dim = 2 * state_dim + action_dim + 1
        self.data_points = jnp.zeros((buffer_size, data_point_dim))
        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size

    def store(self, data_tuple):
        """
        Store datampoint
        :param data_tuple: data of system instance [tuple]
        """
        data_point = jnp.hstack(data_tuple)
        self.data_points = self.data_points.at[self.ptr].set(data_point)
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample_batch(self, batch_size):
        """
        Sample from past instances
        :param batch_size: size of sample batch [int]
        """
        batch_size = min(batch_size, self.size)
        self.rng, rng_input = jrandom.split(self.rng)
        indexes = jrandom.randint(rng_input, shape=(batch_size,),
                                 minval=0, maxval=self.size)
        return self.data_points[indexes]


