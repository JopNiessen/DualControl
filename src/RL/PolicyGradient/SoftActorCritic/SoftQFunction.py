"""

"""

from src.NeuralNetwork.Equinox import SimpleNetwork

import numpy as np

import jax.numpy as jnp
import equinox as eqx
import optax


class SoftQFunction:
    def __init__(self, dimension, key, eta=1e-2):
        self.model = SimpleNetwork(dimension, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.gamma = .9
        self.sample_size = 1
    
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_fn(self, D, value_func):
        bellman_residual = 0
        N = len(D)
        for s0, u, rew, s1 in D:
            Q = self.predict(s0, u)
            Q_hat = rew + self.gamma * value_func(s1)
            bellman_residual += (Q - Q_hat)**2 / 2
        return jnp.mean(bellman_residual / min(N, self.sample_size))
    
    def take_step(self, D, value_func):
        loss, grads = self.loss_fn(D, value_func)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, control, output_value=False, reverse_order=False):
        if reverse_order:
            input = jnp.hstack([control, state])
        else:
            input = jnp.hstack([state, control])
        y_hat = self.model(input)
        if output_value:
            return jnp.mean(y_hat)
        else:
            return y_hat


