"""

"""

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats.norm import logpdf

import equinox as eqx


class PolicyNetwork(eqx.Module):
    """
    Network based on Equinox
    """
    mu_layer: jnp.ndarray
    log_std_layer: jnp.ndarray
    control_lim: jnp.float32

    def __init__(self, in_size, key, control_limit=2):
        """
        Initialize network
        :param in_size: input size [int]
        :param key: PRNGKey
        :param control_limit: min/max control magnitude [int or float]
        """
        keys = jrandom.split(key, 2)
        self.control_lim = control_limit

        self.mu_layer = eqx.nn.Linear(in_size, 1, use_bias=False, key=keys[0])
        self.log_std_layer = eqx.nn.MLP(in_size=in_size, out_size=1, width_size=32, depth=1, key=keys[1])

    def __call__(self, x, key, deterministic=False):
        """
        Forward propagation
        :param x: input
        :param key: PRNGKey
        :param deterministic: boolean indicates if policy is deterministic [bool]
        :return: control [float], log-probability [float]
        """
        mu, std = self.predict(x)

        if deterministic:
            control = mu
        else:
            control = mu + std * jrandom.normal(key, (1,))
        
        #log_prob = -.5 * ((control - mu) / std)**2 - jnp.log(std) + jnp.log(2*jnp.pi)/2
        log_prob = logpdf(control, loc=mu, scale=std)
        log_prob = jnp.clip(log_prob, a_min=-1, a_max=0)

        control = self.control_lim * jnp.tanh(control)
        return control, log_prob
    
    def predict(self, x, squash=False):
        """
        Forward propagation
        :param x: network input [array]
        :param squash: boolean indicates if control is squashed [bool]
        :return: mean [float], standard deviation [float]
        """
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        std = jnp.exp(log_std)
        if squash:
            return jnp.tanh(mu)*self.control_lim, std
        else:
            return mu, std


class QNetwork(eqx.Module):
    """
    Multi-Layer Perceptron
    """
    layer: jnp.ndarray
    
    def __init__(self, in_size, key):
        self.layer = eqx.nn.MLP(in_size=in_size, out_size=1, width_size=32, depth=1, key=key)
    
    def __call__(self, x):
        return self.layer(x)

