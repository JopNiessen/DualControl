"""


"""

from src.NeuralNetwork.Equinox import SimpleNetwork

import numpy as np

import jax.numpy as jnp
import equinox as eqx
import optax


class SoftValueFunction:
    def __init__(self, dimension, key, eta=1e-2):
        self.model = SimpleNetwork(dimension, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.sample_size = 1
    
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_fn(self, D, q_func, pi_log_func):
        squared_residual_error = 0
        N = len(D)
        for s0, u, _, _ in D:
            V = self.predict(s0)
            # Sample u from policy pi
            Q = q_func(s0, u)
            log_pi = pi_log_func(s0, u)
            squared_residual_error += (V - (Q - log_pi))**2 / 2
        return jnp.mean(squared_residual_error / min(N, self.sample_size))

    def take_step(self, D, q_func, pi_log_func):
        loss, grads = self.loss_fn(D, q_func, pi_log_func)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, output_value=False):
        if output_value:
            return jnp.mean(self.model(state))
        else:
            return self.model(state)

