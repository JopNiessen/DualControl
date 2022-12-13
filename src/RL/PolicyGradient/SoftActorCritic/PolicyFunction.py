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
        self.eta = 1e-3
        self.gradient_clipping = .01
    
    def predict(self, state):
        return jnp.dot(self.params, state)
    
    def get_control(self, state):
        u_star = self.predict(state)
        xi = np.random.normal()
        u = u_star + xi * self.stdev
        return u, u_star
    
    def objective(self, params, D, q_func):
        obj = 0
        N_samples = 5
        N = len(D)
        for s0_a, s0_b, _, _, _, _ in D:
            s0 = jnp.array([s0_a, s0_b])
            mu = jnp.dot(params, s0)
            exp_value = 0
            for i in range(N_samples):
                u = mu + np.random.normal(0, self.stdev)
                log_pi = -.5 * ((u - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2
                Q = q_func(s0, u)
                exp_value += (log_pi - Q)
            obj += exp_value / N_samples
            return jnp.mean(obj / N)

    def grad_phi(self, state, control, q_func):
        params = self.params
        grad_phi_log_pi = jax.grad(self.log_pi)(params, state, control)
        grad_u_log_pi = jax.grad(self.log_pi, argnums=2)(params, state, control)
        grad_Q = eqx.filter_grad(q_func)(control, state, output_value=True, reverse_order=True)
        return grad_phi_log_pi + (grad_u_log_pi - grad_Q)*state
    
    def update(self, D, q_func): #state, control, q_func):
        #grads = self.grad_phi(state, control, q_func)
        grads = jax.grad(self.objective)(self.params, D, q_func)
        param_update = self.eta*grads
        norm = jnp.linalg.norm(param_update)
        if norm > self.gradient_clipping:
            param_update = param_update / norm
        self.params += param_update

    def log_pi(self, params, state, control):
        mu = jnp.dot(stop_gradient(params), state)
        prob = -.5 * ((stop_gradient(control) - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2
        return prob[0]

    def log_prob(self, state, control):
        mu = self.predict(state)
        return -.5 * ((control - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2



