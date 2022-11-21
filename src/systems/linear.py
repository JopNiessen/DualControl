"""
2-Dimensional Linear Quadratic (LQ) system with gaussian noise

by J. Niessen
created on: 2022.10.24
"""


import numpy as np
import jax.numpy as jnp
import jax.random as jrandom


# For now, it uses time steps (dt) of 1 sec. With x1 = 'velocity in (m/s)'
class StochasticDoubleIntegrator:
    def __init__(self, x0, b=1, k=.02, dt=.1, time_horizon=4):
        """
        This class describes a 2 dimensional linear dynamical system with gaussian noise

        :param x0: initial state
        :param b: bias term
        :param k: friction
        :param dt: time step size
        :param time_horizon: end time
        """
        self.x = x0
        self.dt = dt
        self.dim = len(x0)

        """System parameters"""
        self.A = jnp.array([[0, 1], [0, -k]])
        self.B = jnp.array([0, b])
        self.C = jnp.identity(self.dim)
        self.v = jnp.array([[0, 0], [0, 0]])
        self.w = jnp.array([[0, 0], [0, .5]])

        """Cost parameters"""
        self.F = jnp.array([[1, 0], [0, 0]])
        self.G = jnp.array([[1, 0], [0, 0]])
        self.R = .5
        self.T = time_horizon

        """Fully observable parameters"""
        self.known_param = {'dim': self.dim, 'A': self.A, 'C': self.C, 'v': self.v, 'w': self.w,
                            'F': self.F, 'G': self.G, 'R': self.R, 'T': self.T}

    def run(self, time):
        x_data = self.x
        time = jnp.arange(0, time, self.dt)
        for _ in time:
            self.update()
            x_data = jnp.vstack((x_data, self.x))
        return time, x_data

    def update(self, key, u=0, info=False):
        """
        Update state (x) according to: x(n+1) = Ax(n) + Bu(n) + Wxi
        :param u: control (zero if not assigned)
        :param info: [boolean] determines if cost is returned
        :return: marginal cost
        """
        x_prev = self.x
        self.x = self.get_state_update(key, self.x, u)
        if info:
            return self.x, self.cost(x_prev, u)

    def get_state_update(self, key, x, u):
        xi = jrandom.normal(key, (self.dim, ))
        #xi = np.random.normal(size=2)
        return x + self.dt * (jnp.dot(self.A, x) + self.B * u) + jnp.sqrt(self.dt) * np.dot(self.w, xi)

    def observe(self, key):
        """
        Observe the state (x) according to: y(n) = Cx(n) + Vxi
        :return: state observation (y)
        """
        xi = jrandom.normal(key, (self.dim, ))
        #xi = np.random.normal(size=2)
        return np.dot(self.C, self.x) + np.dot(self.v, xi)

    def cost(self, x, u=0):
        """
        (Marginal) cost
        :param x: state
        :param u: control
        :return: marginal cost
        """
        return x.T @ self.G @ x + self.R * u**2

    def final_cost(self, x):
        """
        Cost in final timestep (t=T)
        :param x: state
        :return: end cost
        """
        return x.T @ self.F @ x

    def reset(self, x0):
        """
        Reset state
        :param x0: initial state
        """
        self.x = x0



def update_x(A, B, x, u, xi):
    """
    Calculate state (x) in next timestep
    :param A: System matrix
    :param B: Bias
    :param x: state
    :param u: control
    :param xi: noise
    :return: state in next timestep x(n+1)
    """
    return A @ x + B * u + xi


def update_theta(x_old, x_new, theta, u, cov_inv):
    """
    Calculate new belief
    :param x_old: previous state x(n-1)
    :param x_new: current state x(n)
    :param theta: previous belief theta(n-1)
    :param u: last control u(n-1)
    :param cov_inv: inverse of state covariance matrix
    :return: new belief theta(n)
    """
    return theta + (x_new - x_old) * u @ cov_inv

