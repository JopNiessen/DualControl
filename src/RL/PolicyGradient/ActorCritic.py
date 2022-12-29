"""


"""


import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax import grad
from jax.lax import stop_gradient

# local imports
from src.NeuralNetwork.ANN import NeuralNet


def TD_error(V0, V1, reward, gamma=.9):
    """
    Calculate Temporal Differerence (TD) error
    :param V0: Value at t=0 (int)
    :param V1: Value at t=1 (int)
    :param reward: instantaneous reward (int)
    :param gamma: discount factor (int)
    :return: TD error (int)
    """
    Vtarget = TD_target(V1, reward, gamma)
    return Vtarget - V0

def TD_target(V1, reward, gamma=.9):
    """
    """
    return reward + gamma * V1


class Critic:
    def __init__(self, network):
        self.network = network
        self.gamma = .9
    
    def TD_target(self, s1, reward):
        V1 = self.network.predict(s1)
        return reward + self.gamma * V1
    
    def get_value(self, state, learning=False):
        value = self.network.predict(state, learning=learning)
        return value

    def update(self, V0, s1, reward, info=False):
        target = self.TD_target(s1, reward)
        self.network.update(V0, target)
        if info:
            return target - V0


class Actor:
    def __init__(self, key, dim):
        self.params = jrandom.normal(key, dim).T
    
    def get_control(self, state):
        return jnp.dot(self.params, state)
    



def loss(params, state, control, reward):
    mu = jnp.dot(params, state)
    sigma = .5

    pd_value = jnp.exp(-.5*((control - mu) / sigma)**2) / (sigma * jnp.sqrt(2*jnp.pi))

    log_prob = jnp.log(pd_value + 1e-5)

    loss = - reward * log_prob

    return loss

"""
BIN

class Critic:
    def __init__(self, key, dim):
        self.params = jrandom.normal(key, dim).T
        self.lamb = 0
        self.eta = 1e-2
        self.z = jnp.zeros(dim)
    
    def update(self, state, TD):
        grad_V = state
        self.update_eligibility_trace(grad_V)
        self.update_params(TD)

    def update_eligibility_trace(self, grad_V):
        self.z = self.lamb * self.gamma * self.z + grad_V

    def update_params(self, TD):
        self.params += self.eta*TD*self.z
    
    def get_value(self, state):
        return jnp.dot(self.params, state)

"""