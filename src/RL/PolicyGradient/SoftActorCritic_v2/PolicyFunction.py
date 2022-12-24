"""

"""

import jax.numpy as jnp
import jax.random as jrandom
import jax

import equinox as eqx
import optax

from src.RL.PolicyGradient.SoftActorCritic_v2.NeuralNets import PolicyNetwork


class SoftPolicyFunction:
    def __init__(self, in_size, key, eta=1e-2):
        self.model = PolicyNetwork(in_size, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
    
    def loss_fn(self, model, D, q_func, keys):
        control, log_prob = jax.vmap(model)(D, keys)
        q_value = jax.vmap(q_func)(D, control)
        return jnp.mean(log_prob - q_value)

    def update(self, D, q_func, key):
        keys = jrandom.split(key, len(D))
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, D, q_func, keys)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)

    def get_control(self, state, key):
        control, log_prob = self.model(state, key)
        return control, log_prob

