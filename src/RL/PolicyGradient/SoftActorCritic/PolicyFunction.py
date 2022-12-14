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
        #self.model = SimpleNetwork((n_states, 32, n_controls*2), key)
        self.model = ManualNetwork((n_states, n_controls*2), ('none'), key) # linear model
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)

        # create manual function
        self.grad = eqx.filter_value_and_grad
   
    def sample_control(self, mu, sigma, key):
        """
        Sample control
        :param mu: optimal control (network output)
        :param sigma: standard deviation (network output)
        :param key: PRNGKey
        :return: sampled control
        :return: log probability
        """
        control = mu + jrandom.normal(key, (1,)) * sigma
        log_prob = -.5 * ((control - mu) / sigma)**2 - jnp.log(sigma) + jnp.log(2*jnp.pi)/2
        return control, log_prob

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
        output = jax.vmap(model)(D)
        mu = output[:,0]
        sigma = output[:,1]
        control, log_prob = self.sample_control(mu, sigma, key)
        q_value = jax.vmap(q_func)(D, control)
        loss = jnp.mean(log_prob - q_value)
        
        #entropy = None

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
    
    def log_prob(self, state, control, vmap=True):
        """
        Log probability given control Log p(control|state)
        :param state: state
        :param control: control
        :param vmap: indicate if function should use vector mapping [boolean]
        :return: log probability [float]
        """
        if vmap:
            output = jax.vmap(self.model)(state)
            mu = output[:,0]
            sigma = output[:,1]
            return self.log_prob_fn(control[:,0], mu, sigma)
        else:
            mu, sigma = self.model(state)
            return self.log_prob_fn(control, mu, sigma)
    
    def log_prob_fn(self, x, mu, sigma):
        return -.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) + jnp.log(2*jnp.pi)/2

    def get_control(self, state, key):
        """
        Fetch control
        :param state: state
        :param key: PRNGKey
        :return: sampled control
        :return: optimal control
        """
        (mu, sigma) = self.model(state)
        control = mu + jrandom.normal(key, (1,))*sigma
        return control, mu


