"""


"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax import grad
from jax.lax import stop_gradient


class linear_controller:
    def __init__(self, key, dim, eta=1e-3):
        """
        REINFORCE algorithm with Gaussian policy and continuous control space
        :param key: jax PRNGKey
        :param dim: (input, output)
        :param eta: learning rate
        """
        self.params = jrandom.normal(key, dim)
        
        self.eta = eta
        self.stdev = .5

        self.gamma = .9
        self.eta = eta

        self.RH = 5
        self.n_traj = 1

        self.it = 0
        self.G = 0
        self.dtheta = 0
        self.update_interval = 5
    
    def update(self, y0, u_star, reward, done):
        self.G += self.gamma*self.G + reward
        grad_pi = self.grad_pi(u_star, y0)
        self.dtheta += grad_pi*self.G
        self.it += 1
        if self.it == self.update_interval:
            self.param_update(self.dtheta)
            self.G = 0
            self.dtheta = 0
            self.it = 0
    
    def get_control(self, x, optimal=False):
        """
        Get control value (u)
        :param x: state/observation [array]
        :param optimal: [booleam]
        :return: control value
        """
        if optimal:
            u = jnp.dot(self.params, x)[0]
        else:
            mu = jnp.dot(self.params, x)[0]
            xi = np.random.normal()
            u = mu + xi * self.stdev
        return u
    
    def param_update(self, grad):
        """
        Update parameters
        :param params: learned parameter values
        """
        dparams = self.eta*grad
        norm = jnp.linalg.norm(dparams)
        if norm >= .1:
            dparams = .1 * dparams / norm
        self.params -= dparams
    
    def pi(self, params, u, s):
        """
        Gaussian policy function
        :param params: policy parameters
        :param u: control value
        :param s: state/observation [array]
        :return: probability p(u|P, s)
        """
        P = stop_gradient(params)
        mu = jnp.dot(P, s)[0]
        prob = jnp.exp(-.5*((mu - u) / self.stdev)**2) / (self.stdev * jnp.sqrt(2*jnp.pi))
        return prob
    
    def grad_pi(self, u, s):
        """
        Gradient of gaussian policy function
        :param u: control value
        :param s: state/observation [array]
        :return: gradient pi wrt parameters
        """
        mu = jnp.dot(self.params, s)
        return (u - mu)*s / self.stdev
    
    def run_simulation(self, stm):
        """
        
        """
        key = jrandom.PRNGKey(0)
        x0 = stm.x
        dtheta = 0
        for _ in range(self.n_traj):
            stm.reset(x0)
            x = x0
            G = 0
            for _ in range(self.RH):
                y0 = stm.observe(key)
                u = self.get_control(y0)
                rew = -stm.cost(x, u)

                G += self.gamma*G + rew
                params = self.params
                #grad_pi = grad(self.pi)(params, u, y0) #equivalent result, but probably slower
                grad_pi = self.grad_pi(u, y0)
                dtheta += grad_pi*G
                stm.update(key, u)
                key, subkey = jrandom.split(key)
            self.param_update(dtheta)



class DeepREINFORCE:
    def __init__(self, DeepNet, eta=1e-3):
        """
        REINFORCE algorithm with Gaussian policy and continuous control space
        :param key: jax PRNGKey
        :param dim: (input, output)
        :param eta: learning rate
        """
        self.network = DeepNet
        
        self.eta = eta
        self.stdev = .5

        self.gamma = .9
        self.eta = eta

        self.RH = 5
        self.n_traj = 1

        self.it = 0
        self.G = 0
        self.dtheta = 0
        self.update_interval = 5
    
    def update(self, y0, u_star, reward, done):
        self.G += self.gamma*self.G + reward
        grad_pi = self.grad_pi(u_star, y0)
        self.dtheta += grad_pi*self.G
        self.it += 1
        if self.it == self.update_interval:
            norm = jnp.linalg.norm(self.dtheta)
            if norm > 1:
                self.dtheta = self.dtheta / norm
            self.network.train_step(y0, self.dtheta)
            self.G = 0
            self.dtheta = 0
            self.it = 0
    
    def get_control(self, x, optimal=False):
        """
        Get control value (u)
        :param x: state/observation [array]
        :param optimal: [booleam]
        :return: control value
        """
        u = self.network.predict(x)[0]
        if not optimal:
            xi = np.random.normal()
            u += xi * self.stdev
        return u
    
    def param_update(self, grad):
        """
        Update parameters
        :param params: learned parameter values
        """
        dparams = self.eta*grad
        norm = jnp.linalg.norm(dparams)
        if norm >= .1:
            dparams = .1 * dparams / norm
        self.params -= dparams
    
    def pi(self, u, s):
        """
        Gaussian policy function
        :param params: policy parameters
        :param u: control value
        :param s: state/observation [array]
        :return: probability p(u|P, s)
        """
        mu = self.network.predict(s)[0]
        prob = jnp.exp(-.5*((mu - u) / self.stdev)**2) / (self.stdev * jnp.sqrt(2*jnp.pi))
        return prob
    
    def grad_pi(self, u, s):
        """
        Gradient of gaussian policy function
        :param u: control value
        :param s: state/observation [array]
        :return: gradient pi wrt parameters
        """
        mu = self.network.predict(s)
        return (u - mu)*s / self.stdev

