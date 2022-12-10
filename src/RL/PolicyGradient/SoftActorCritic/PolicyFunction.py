"""


"""

from src.NeuralNetwork.Equinox import SimpleNetwork

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.lax import stop_gradient

import equinox as eqx



class SoftPolicyFunction:
    def __init__(self, dim, key):
        self.params = jrandom.normal(key, dim).T
        self.stdev = .5
        self.eta = 1e-2
    
    def predict(self, state):
        return jnp.dot(self.params, state)
    
    def get_control(self, state):
        u_star = self.predict(state)
        xi = np.random.normal()
        u = u_star + xi * self.stdev
        return u, u_star
    
    def grad_phi(self, state, control, q_func):
        params = self.params
        grad_phi_log_pi = jax.grad(self.log_pi)(params, state, control)
        grad_u_log_pi = jax.grad(self.log_pi, argnums=2)(params, state, control)
        grad_Q = eqx.filter_grad(q_func)(control, state, output_value=True, reverse_order=True)
        return grad_phi_log_pi + (grad_u_log_pi - grad_Q)*state
    
    def update(self, state, control, q_func):
        grads = self.grad_phi(state, control, q_func)
        self.params += self.eta * grads

    def log_pi(self, params, state, control):
        mu = jnp.dot(stop_gradient(params), state)
        prob = -.5 * ((stop_gradient(control) - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2
        return prob[0]

    def log_prob(self, state, control):
        mu = self.predict(state)
        return -.5 * ((control - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2



