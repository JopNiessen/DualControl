"""

"""

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom
import jax

import equinox as eqx
import optax

# import local libaries
from src.NeuralNetwork.Equinox import SimpleNetwork


class SoftQFunction:
    """
    Soft Q-function
    """
    def __init__(self, in_size, key, eta=1e-2):
        """
        :param in_size: input size [int]
        :param key: PRNGKey
        :param eta: learning rate [float]
        """
        self.model = SimpleNetwork((in_size, 32, 1), key)
        #self.model = QNetwork(in_size, key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)
        self.gamma = .9
    
    def loss_fn(self, model, full_state, q_target):
        """
        Bellman residual loss
        :param model: Q-model [NN]
        :param full_state: observations and control [array]
        :param q_target: target q-value [float]
        :return: bellman residual loss [float]
        """
        q0_hat = jax.vmap(model)(full_state)
        bellman_residual = jnp.mean((q0_hat - q_target)**2 / 2)
        return bellman_residual
    
    def update(self, full_state, reward, next_state, policy, key):
        """
        Update Q-network
        :param full_state: observations and control [array]
        :param reward: rewards [array]
        :param next_state: observations in next timestep [array]
        :param policy: policy model
        :param key: PRNGKey
        :return: bellman residual loss [float]
        """
        # Estimate target q-value
        keys = jrandom.split(key, len(full_state))
        control, entropy = jax.vmap(policy.get_control)(next_state, keys)
        full_state1 = jnp.hstack([next_state, control])
        q1_hat = jax.vmap(self.model)(full_state1)
        V1 = q1_hat - entropy
        q_target = reward + self.gamma * V1

        # Update q-network
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(self.model, full_state, q_target)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def predict(self, state, control):
        """
        Predict q-value
        :param state: state [array]
        :param control: control [float]
        :return: predicted q-value [float]
        """
        input = jnp.hstack([state, control])
        return self.model(input)

