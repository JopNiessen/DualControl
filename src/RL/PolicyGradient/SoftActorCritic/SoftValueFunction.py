"""
Soft Actor-Critic: Value function
"""

# import local libraries
from src.NeuralNetwork.Equinox import SimpleNetwork

# import global libraries
import jax
import jax.numpy as jnp
import equinox as eqx
import optax


class SoftValueFunction:
    """
    Soft value-function
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

        # create jit function
        self.grad = eqx.filter_value_and_grad
        #self.predict = eqx.filter_jit(self._predict)
    
    @eqx.filter_jit
    def loss_fn(self, model, D, q_func, pi_log_func, get_control, key):
        """
        Calculate squared residual error
        :input model: Value-network
        :input D: Replay buffer
        :input q_func: Q-function [function]
        :input pi_log_func: log P(control|state) [function]
        :input get_control: policy function that samples a control [function]
        :input key: PRNGKey
        :return: loss
        """
        squared_residual_error = 0
        N = len(D)
        # loops over replay buffer
        for s0_a, s0_b, u, _, _, _ in D:
            s0 = jnp.array([s0_a, s0_b])
            V = model(s0)
            #u, _ = get_control(s0, key)
            Q = q_func(s0, u)
            log_pi = pi_log_func(s0, u)
            squared_residual_error += (V - (Q - log_pi))**2 / 2
        return jnp.mean(squared_residual_error / N)

    def take_step(self, D, q_func, pi_log_func, get_control, key):
        """
        Update Value-network parameters
        :input D: Replay buffer
        :input q_func: Q-function [function]
        :input pi_log_func: log P(control|state) [function]
        :input get_control: policy function that samples a control [function]
        :input key: PRNGKey
        :return: loss
        """
        loss, grads = self.grad(self.loss_fn)(self.model, D, q_func, pi_log_func, get_control, key)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, output_value=False):
        """
        Estimate state-value
        :input state: state
        :input output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :return: state-value [float] or [DeviceArray]
        """
        if output_value:
            return jnp.mean(self.model(state))
        else:
            return self.model(state)

