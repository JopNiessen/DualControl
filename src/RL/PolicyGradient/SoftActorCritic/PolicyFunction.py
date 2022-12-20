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

from jax.scipy.stats.norm import logpdf


class PolicyNetwork(eqx.Module):
    """
    Network based on Equinox
    """
    layers: list
    mu_layer: jnp.ndarray
    log_std_layer: jnp.ndarray
    control_lim: jnp.float32

    def __init__(self, dim, activation, key, control_limit=4):
        """
        Initialize network
        :param dim: network dimensions (n_input, ..., n_output)
        :param key: PRNGKey
        """
        self.layers = []
        self.control_lim = control_limit
        n_input, n_hidden = 2, 32

        N = len(dim)
        key, subkey = jrandom.split(key)
        self.mu_layer = eqx.nn.Linear(dim[-2], 1, key=key)
        self.log_std_layer = eqx.nn.Linear(dim[-2], 1, key=key)
        #self.log_std_layers = [eqx.nn.Linear(dim[-2], n_hidden, key=subkey),
        #                        jax.nn.relu,
        #                        eqx.nn.Linear(n_hidden, dim[-1], key=subkey)]
        for idx in range(N-1):
            key, _ = jrandom.split(key)
            self.layers.append(eqx.nn.Linear(dim[idx], dim[idx+1], key=key))
            self.add_activation_fun(activation[idx])
        #self.layers.append(eqx.nn.Linear(dim[-2], dim[-1], key=key))
        #self.bias = jnp.ones(dim[-1])
    
    def add_activation_fun(self, activ):
        if activ == 'relu':
            self.layers.append(jax.nn.relu)
        else:
            pass

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
        
        log_prob = logpdf(control, loc=mu, scale=std)
        log_prob -= (2*(jnp.log(2) - control - jax.nn.softplus(-2*control)))
        
        control = self.control_lim * jnp.tanh(control)
        return control, log_prob
    
    def predict(self, x):
        """
        Forward propagation
        :param x: input
        :return: network output
        """
        for layer in self.layers:
            x = layer(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        #for layer in self.log_std_layers:
        #    log_std = layer(x)
        std = jnp.exp(log_std)
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
        #self.model = SimpleNetwork((n_states, 32, n_controls*2), key)
        #self.model = ManualNetwork((n_states, n_controls*2), ('none'), key) # linear model
        self.model = PolicyNetwork((n_states, n_controls*2), ('none'), key)
        self.optimizer = optax.adam(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.stdev = .5
        self.alpha = 0

        # create manual function
        self.grad = eqx.filter_value_and_grad
   
    def sample_control(self, D, key):
        """
        Sample control
        """
        control, log_prob = jax.vmap(self.model)(D, key)
        return control, log_prob
    
    def sample_regularizerd_control(self, mu, sigma, key):
        sigma = self.stdev
        xi = jrandom.normal(key, (1,))
        control = jnp.tanh(mu + sigma*xi)
        log_prob = self.log_prob_fn(control, mu, sigma)
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
        key = jrandom.split(key, len(D))
        control, log_prob = jax.vmap(model)(D, key)
        #output = jax.vmap(model)(D)
        #mu = output[:,0]
        #sigma = output[:,1]
        #control, log_prob = self.sample_control(mu, sigma, key)
        #control, log_prob = self.sample_regularizerd_control(mu, sigma, key)
        q_value = jax.vmap(q_func)(D, control)
        loss = jnp.mean(self.alpha*log_prob - q_value)
        
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
            mu, std = jax.vmap(self.model.predict)(state)
        else:
            mu, std = self.model.predict(state)
        log_prob = logpdf(control, loc=mu, scale=std)
        log_prob -= (2*(jnp.log(2) - control - jax.nn.softplus(-2*control)))
        return self.alpha*log_prob
            #mu = output[:,0]
            #sigma = output[:,1]
            #return self.log_prob_fn(control[:,0], mu, sigma)
        #else:
            #mu, sigma = self.model(state)
            #return self.log_prob_fn(control, mu, sigma)
    
    def log_prob_fn(self, x, mu, sigma):
        sigma = self.stdev
        return -.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) + jnp.log(2*jnp.pi)/2

    def get_control(self, state, key, deterministic=False):
        """
        Fetch control
        :param state: state
        :param key: PRNGKey
        :return: sampled control
        :return: optimal control
        """
        control, log_std = self.model(state, key, deterministic=deterministic)
        sigma = jnp.exp(log_std)
        #sigma = self.stdev
        #if noise:
        #    control = jnp.tanh(mu + jrandom.normal(key, (1,))*sigma)
        #else:
        #    control = jnp.tanh(mu)
        return control, sigma


