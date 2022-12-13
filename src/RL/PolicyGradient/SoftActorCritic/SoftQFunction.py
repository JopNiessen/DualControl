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
    
    @eqx.filter_jit
    def loss_fn(self, model, D, value_func):
        bellman_residual = 0
        N = len(D)
        for s0_a, s0_b, u, rew, s1_a, s1_b in D:
            input = jnp.array([s0_a, s0_b, u])
            s1 = jnp.array([s1_a, s1_b])
            #input = jnp.hstack([s0, u])
            Q = model(input)
            Q_hat = rew + self.gamma * value_func(s1)
            bellman_residual += (Q - Q_hat)**2 / 2
        return jnp.mean(bellman_residual / N)
    
    def take_step(self, D, value_func):
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, D, value_func)
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


