"""
Soft Actor-Critic: Q-function
"""

# import local libraries
from src.NeuralNetwork.Equinox import SimpleNetwork

# import grobal libraries
import jax.numpy as jnp
import equinox as eqx
import optax


class SoftQFunction:
    """
    Soft Q-network
    """
    def __init__(self, dimension, key, eta=1e-2):
        """
        Initialize network
        :param dim: network dimensions (n_inputs, n_hidden, n_output)
        :param key: PRNGKey
        :param eta: learning rate
        """
        self.model = SimpleNetwork(dimension, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.gamma = .9
    
        # create jit function
        self.grad = eqx.filter_value_and_grad
    
    @eqx.filter_jit
    def loss_fn(self, model, D, value_func):
        """
        Calculate bellman residual loss
        :input model: Q-network
        :input D: Replay buffer
        :input value_func: value function [function]
        :return: loss
        """
        bellman_residual = 0
        N = len(D)
        # Loops over replay buffer
        for s0_a, s0_b, u, rew, s1_a, s1_b in D:
            input = jnp.array([s0_a, s0_b, u])
            s1 = jnp.array([s1_a, s1_b])
            Q = model(input)
            Q_hat = rew + self.gamma * value_func(s1)
            bellman_residual += (Q - Q_hat)**2 / 2
        return jnp.mean(bellman_residual / N)
    
    #@eqx.filter_jit
    def take_step(self, D, value_func):
        """
        Update Q-network parameters
        :input D: replay buffer
        :input value_func: value function [function]
        :return loss
        """
        loss, grads = self.grad(self.loss_fn)(self.model, D, value_func)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, control, output_value=False, reverse_order=False):
        """
        Estimate Q-value
        :input state: state
        :input control: control
        :input output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :input reverse_order: switch state-control order [boolean]
        :return: Q-value [float] or [DeviceArray]
        """
        if reverse_order:
            input = jnp.hstack([control, state])
        else:
            input = jnp.hstack([state, control])
        y_hat = self.model(input)
        if output_value:
            return jnp.mean(y_hat)
        else:
            return y_hat


