"""

"""

import jax.numpy as jnp
import jax.random as jrandom
import jax

import equinox as eqx
import optax

from src.NeuralNetwork.Equinox import SimpleNetwork


class SoftQFunction:
    def __init__(self, in_size, key, eta=1e-2):
        self.model = SimpleNetwork((in_size, 32, 1), key)
        #self.model = QNetwork(in_size, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.gamma = .9
    
    def update(self, full_state, reward, next_state, policy, key):
        keys = jrandom.split(key, len(full_state))
        control, log_pi = jax.vmap(policy.get_control)(next_state, keys)
        full_state1 = jnp.hstack([next_state, control])
        q1_hat = jax.vmap(self.model)(full_state1)
        q_target = reward + self.gamma * q1_hat

        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, full_state, q_target)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def loss_fn(self, model, full_state, q_target):
        q0_hat = jax.vmap(model)(full_state)
        bellman_residual = jnp.mean((q0_hat - q_target)**2 / 2)
        return bellman_residual
    
    def predict(self, state, control):
        input = jnp.hstack([state, control])
        return self.model(input)

