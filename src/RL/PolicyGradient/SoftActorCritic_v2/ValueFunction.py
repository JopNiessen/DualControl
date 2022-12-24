"""


"""

import jax.numpy as jnp
import jax.random as jrandom
import jax

import equinox as eqx
import optax

from src.NeuralNetwork.Equinox import SimpleNetwork


class SoftValueFunction:
    """
    Soft value-function
    """
    def __init__(self, in_size, key, eta=1e-2):
        """
        Initialize network
        :param dim: network dimensions (n_inputs, n_hidden, n_output)
        :param key: PRNGKey
        :param eta: learning rate
        """
        self.model = SimpleNetwork((in_size, 32, 1), key)
        #self.model = QNetwork(in_size, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
    
    def loss_fn(self, model, D_state, V_target):
        """
        Calculate squared residual error
        :param model: Value-network
        :param D_state: replay buffer (state values)
        :param D_control: replay buffer (control values)
        :param q_func: Q-function [function]
        :param pi_log_func: log P(control|state) [function]
        :param get_control: policy function that samples a control [function]
        :param key: PRNGKey
        :return: loss
        """
        V_hat = jax.vmap(model)(D_state)
        residual_error = jnp.mean((V_hat - V_target)**2 / 2)
        return residual_error

    def update(self, D_state, q_func, policy, key):
        """
        Update Value-network parameters
        :param D_state: replay buffer (state values)
        :param D_control: replay buffer (control values)
        :param q_func: Q-function [function]
        :param get_control: policy function that samples a control [function]
        :param key: PRNGKey
        :return: loss
        """
        D_control, log_pi = jax.vmap(policy.get_control)(D_state, jrandom.split(key, len(D_state)))
        V_target = jax.vmap(q_func)(D_state, D_control) - log_pi
        
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, D_state, V_target)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, output_value=False):
        """
        Estimate state-value
        :param state: state
        :param output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :return: state-value [float] or [DeviceArray]
        """
        if output_value:
            return jnp.mean(self.model(state))
        else:
            return self.model(state)