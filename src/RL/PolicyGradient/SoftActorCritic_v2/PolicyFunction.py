"""

"""

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom
import jax

import equinox as eqx
import optax

# import local libraries
from src.RL.PolicyGradient.SoftActorCritic_v2.NeuralNets import PolicyNetwork


class SoftPolicyFunction:
    """
    Soft Actor-Critic policy function
    """
    def __init__(self, in_size, key, eta=1e-2, alpha=0):
        """
        :param in_size: input size [int]
        :param key: PRNGKey
        :param eta: learning rate [float]
        :param alpha: entropy regularization coefficient [float]
        """
        self.model = PolicyNetwork(in_size, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.alpha = alpha
    
    def loss_fn(self, model, D, q_func, keys):
        """
        loss function
        :param model: policy model [NN]
        :param D: observations [array]
        :param q_func: Q-function
        :param keys: list of PRNGKeys
        :return: loss
        """
        control, log_prob = jax.vmap(model)(D, keys)
        q_value = jax.vmap(q_func)(D, control)
        return jnp.mean(self.alpha * log_prob - q_value)

    def update(self, D, q_func, key):
        """
        Update policy
        :param D: observations [array]
        :param q_func: Q-functions
        :param key: PRNGKey
        :return: loss
        """
        keys = jrandom.split(key, len(D))
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, D, q_func, keys)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss

    def get_control(self, state, key):
        """
        Predict control
        :param state: state [array]
        :param key: PRNGKey
        :return: control [float], entropy [float]
        """
        control, log_prob = self.model(state, key)
        return control, self.alpha*log_prob

