"""


"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import jax
from jax.lax import stop_gradient

from src.NeuralNetwork.ANN import NeuralNet
from src.utilities.utilities import Tracker


class SoftActorCritic:
    def __init__(self, key, q_dim, q_act, v_dim, v_act, pi_dim):
        self.SQF = SoftQFunction(key, q_dim, q_act)
        self.SVF = SoftValueFunction(key, v_dim, v_act)
        self.PI = SoftPolicyFunction(key, pi_dim)
        self.buffer = list()
        self.tracker = Tracker(['state0', 'state1', 'control', 'cost', 'V_value', 'V_loss', 
                                    'Q_value', 'Q_loss', 'policy_angle', 'policy_force'])
    
    def update(self, s0, u, tracking=True):
        v_value = self.SVF.predict(s0, learning=True)[0, 0]
        q_value = self.SQF.predict(s0, u, learning=True)[0, 0]
        v_loss = self.SVF.update(self.buffer, self.SQF.predict, self.PI.log_prob)[0, 0]
        q_loss = self.SQF.update(self.buffer, self.SVF.predict)[0, 0]

        self.PI.update(s0, u, self.SQF.predict)

        if tracking:
            control_angle = jnp.arctan2(self.PI.params[0,0], self.PI.params[0,1])
            control_force = jnp.linalg.norm(self.PI.params)
            self.tracker.add([s0[0], s0[1], u, None, v_value, v_loss, q_value, q_loss,
                                    control_angle, control_force])
    
    def get_control(self, state):
        return self.PI.get_control(state)
    
    def add_to_buffer(self, transition):
        self.buffer.append(transition)


class SoftQFunction(NeuralNet):
    def __init__(self, key, dimension, activation):
        super().__init__(key, dimension, activation)
        self.gamma = .9
        self.sample_size = 10
    
    def loss(self, D, value_func):
        bellman_residual = 0
        N = len(D)
        for it in range(min(N, self.sample_size)):
            if it == 0:
                s0, u, rew, s1 = D[-1]
            else:
                idx = np.random.randint(0, N)
                s0, u, rew, s1 = D[idx]
            Q = self.predict(s0, u)
            Q_hat = rew + self.gamma * value_func(s1)
            bellman_residual += (Q - Q_hat)**2 / 2
        return bellman_residual / min(N, self.sample_size)
    
    def update(self, D, value_func):
        loss = self.loss(D, value_func)
        grads = self.backpropagation(loss)
        self.update_params(grads)
        return loss
    
    def predict(self, state, control, learning=False, value_output=False):
        input = jnp.hstack([state, control])
        y_hat = self.forward_propagation(input, learning=learning)
        if value_output:
            return y_hat[0,0]
        else:
            return y_hat


class SoftValueFunction(NeuralNet):
    def __init__(self, key, dimension, activation):
        super().__init__(key, dimension, activation)
        self.sample_size = 10
    
    def loss(self, D, q_func, pi_log_func):
        squared_residual_error = 0
        N = len(D)
        for it in range(min(N, self.sample_size)):
            if it == 0:
                s0, u, _, _ = D[-1]
            else:
                idx = np.random.randint(0, N)
                s0, u, _, _ = D[idx]
            V = self.predict(s0)
            # Sample u from policy pi
            Q = q_func(s0, u)
            log_pi = pi_log_func(s0, u)
            squared_residual_error += (V - (Q - log_pi))**2 / 2
        return squared_residual_error / min(N, self.sample_size)

    def update(self, D, q_func, pi_log_func):
        loss = self.loss(D, q_func, pi_log_func)
        grads = self.backpropagation(loss)
        self.update_params(grads)
        return loss


class SoftPolicyFunction:
    def __init__(self, key, dim):
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
        grad_Q = jax.grad(q_func, argnums=1)(state, control, value_output=True)
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



