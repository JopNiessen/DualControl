"""
Soft Actor-Critic: Policy function
"""

# import local libraries
from src.NeuralNetwork.Equinox import *

# import global libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax


class PolicyNetwork(eqx.Module):
    """
    Network based on Equinox
    """
    mu_layer: jnp.ndarray
    #std_layers: list
    log_std_layer: jnp.ndarray
    control_lim: jnp.float32
    alpha: jnp.float32

    def __init__(self, dim, key, control_limit=1):
        """
        Initialize network
        :param dim: network dimensions (n_input, ..., n_output)
        :param key: PRNGKey
        """
        key0, key1, key2, key3 = jrandom.split(key, 4)
        #self.std_layers = [eqx.nn.Linear(dim[0], 32, key=key2),
        #                    jax.nn.relu,
        #                    eqx.nn.Linear(32, 1, key=key3)]
        self.control_lim = control_limit
        self.alpha = 0

        self.mu_layer = eqx.nn.Linear(dim[0], 1, key=key0)
        self.log_std_layer = eqx.nn.Linear(dim[0], 1, key=key1)

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
        #for layer in self.std_layers:
        #    x = layer(x)
        #log_std = x
        log_std = self.log_std_layer(x)
        std = jnp.exp(log_std)
        if squash:
            return jnp.tanh(mu), std
        else:
            return mu, std


class SoftPolicyFunction:
    """
    Policy function
    """
    def __init__(self, dim, key, eta=1e-3):
        """
        Initialize network
        :param dim: network dimensions (n_inputs, n_hidden, n_output)
        :param key: PRNGKey
        :param eta: learning rate
        """
        n_states, n_controls = dim
        self.model = PolicyNetwork((n_states, n_controls), key)
        self.optimizer = optax.adam(eta)
        self.opt_state = self.optimizer.init(self.model)

        # create manual function
        self.grad = eqx.filter_value_and_grad
   
    #@eqx.filter_jit
    def loss_fn(self, model, D, q_func, key):
        """
        Calculate loss
        :param model: policy network
        :param D: replay buffer
        :param q_func: Q-function [function]
        :param key: PRNGKey
        :return: loss
        """
        key = jrandom.split(key, len(D))
        control, log_prob = jax.vmap(model)(D, key)
        q_value = jax.vmap(q_func)(D, control)
        loss = jnp.mean(log_prob - q_value)
        
        # Adding regularization
        #l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss

    def take_step(self, D, q_func, key):
        """
        Update policy network parameters
        :param D: replay buffer (only state values)
        :param q_func: Q-function [function]
        :param key: PRNGKey
        :return: loss
        """
        loss, grads = self.grad(self.loss_fn)(self.model, D, q_func, key)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss

    def get_control(self, state, key, deterministic=False):
        """
        Fetch control
        :param state: state
        :param key: PRNGKey
        :return: sampled control
        :return: optimal control
        """
        control, log_prob = self.model(state, key, deterministic=deterministic)
        return control, log_prob


