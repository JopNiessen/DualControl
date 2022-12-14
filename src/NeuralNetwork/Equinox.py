"""
Neural Network based on Equinox
"""

# import global libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


class SimpleNetwork(eqx.Module):
    """
    Network based on Equinox
    > Two hidden layers
    > ReLU activated
    """
    layers: list
    bias: jnp.ndarray

    def __init__(self, dim, key):
        """
        Initialize network
        :param dim: network dimensions (n_input, n_hidden, n_output)
        :param key: PRNGKey
        """
        in_size, hidden_size, out_size = dim
        key1, key2, key3 = jrandom.split(key, 3)
        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                        jax.nn.relu,
                        eqx.nn.Linear(hidden_size, hidden_size, key=key2),
                        jax.nn.relu,
                        eqx.nn.Linear(hidden_size, out_size, key=key3)]
        self.bias = jnp.ones(out_size)
    
    def __call__(self, x):
        """
        Forward propagation
        :param x: input
        :return: network output
        """
        for layer in self.layers:
            x = layer(x)
        return x + self.bias


class ManualNetwork(eqx.Module):
    """
    Network based on Equinox
    """
    layers: list
    bias: jnp.ndarray

    def __init__(self, dim, activation, key):
        """
        Initialize network
        :param dim: network dimensions (n_input, n_hidden, n_output)
        :param key: PRNGKey
        """
        self.layers = []
        N = len(dim)
        for idx in range(N-1):
            in_size = dim[idx]
            out_size = dim[idx+1]
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=key))
            self.add_activation_fun(activation[idx])
            key, _ = jrandom.split(key)
        self.layers.append(eqx.nn.Linear(dim[-2], dim[-1], key=key))
        self.bias = jnp.ones(dim[-1])
    
    def add_activation_fun(self, activ):
        if activ == 'relu':
            self.layers.append(jax.nn.relu)
        else:
            pass

    def __call__(self, x):
        """
        Forward propagation
        :param x: input
        :return: network output
        """
        for layer in self.layers:
            x = layer(x)
        return x + self.bias
