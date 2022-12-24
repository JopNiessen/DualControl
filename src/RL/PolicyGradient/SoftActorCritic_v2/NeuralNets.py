"""

"""

import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


class PolicyNetwork(eqx.Module):
    """
    Network based on Equinox
    """
    mu_layer: jnp.ndarray
    log_std_layer: jnp.ndarray
    control_lim: jnp.float32
    alpha: jnp.float32

    def __init__(self, in_size, key, control_limit=1):
        """
        Initialize network
        :param dim: network dimensions (n_input, ..., n_output)
        :param key: PRNGKey
        """
        key0, key1, key2, key3 = jrandom.split(key, 4)
        self.control_lim = control_limit
        self.alpha = 0

        self.mu_layer = eqx.nn.Linear(in_size, 1, use_bias=False, key=key0)
        self.log_std_layer = eqx.nn.MLP(in_size=in_size, out_size=1, width_size=32, depth=1, key=key1)

    def __call__(self, x, key, deterministic=False):
        """
        Forward propagation
        :param x: input
        :return: network output
        """
        mu, std = self.predict(x)

        if deterministic:
            control = mu
        else:
            control = mu + std * jrandom.normal(key, (1,))
        
        log_prob = -.5 * ((control - mu) / std)**2 - jnp.log(std) + jnp.log(2*jnp.pi)/2
        
        control = self.control_lim * jnp.tanh(control)
        return control, self.alpha*log_prob
    
    def predict(self, x, squash=False):
        """
        Forward propagation
        :param x: input
        :return: network output
        """
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        std = jnp.exp(log_std)
        if squash:
            return jnp.tanh(mu), std
        else:
            return mu, std


class QNetwork(eqx.Module):
    layer: jnp.ndarray
    
    def __init__(self, in_size, key):
        self.layer = eqx.nn.MLP(in_size=in_size, out_size=1, width_size=32, depth=1, key=key)
    
    def __call__(self, x):
        return self.layer(x)

