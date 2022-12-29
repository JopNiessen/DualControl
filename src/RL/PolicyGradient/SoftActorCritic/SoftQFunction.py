"""
Soft Actor-Critic: Q-function
"""

# import local libraries
from src.NeuralNetwork.Equinox import SimpleNetwork

# import grobal libraries
import jax
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
        self.gamma = .1
    
        # create manual functions
        self.grad = eqx.filter_value_and_grad
    
    #@eqx.filter_jit
    def loss_fn(self, model, D_full_state0, q_target):
        """
        Calculate bellman residual loss
        :param model: Q-network
        :param D_full_state0: replay buffer (state & control values)
        :param D_reward: replay buffer (reward values)
        :param D_state1: replay buffer (new state value)
        :param value_func: value function [function]
        :return: loss
        """
        q_hat = jax.vmap(model)(D_full_state0)
        bellman_residual = jnp.mean((q_hat - q_target)**2 / 2)
        return bellman_residual
    
    def take_step(self, D_full_state0, D_reward, D_state1, value_func):
        """
        Update Q-network parameters
        :param D_full_state0: replay buffer (state & control values)
        :param D_reward: replay buffer (reward values)
        :param D_state1: replay buffer (new state value)
        :param value_func: value function [function]
        :return loss
        """
        q_target = D_reward[:,0] + self.gamma * jax.vmap(value_func)(D_state1)

        loss, grads = self.grad(self.loss_fn)(self.model, D_full_state0, q_target)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, control, output_value=False, reverse_order=False):
        """
        Estimate Q-value
        :param state: state
        :param control: control
        :param output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :param reverse_order: switch state-control order [boolean]
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


