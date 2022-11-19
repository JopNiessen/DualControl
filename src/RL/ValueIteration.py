"""


"""

import numpy as np
# from scipy.interpolate import barycentric_interpolate
# from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline


class ValueIteration:
    def __init__(self, alpha=.1, gamma=.9, nbins=(11, 21)):
        self.alpha = alpha
        self.gamma = gamma

        self.nbins = nbins
        self.state_n = len(self.nbins)
        self.cntr_space = np.array([-1, 1])
        self.cntr_n = len(self.cntr_space)

        self.state_amp = np.array([3, 4])

        # Discrete state-space
        self.s0 = np.linspace(-1, 1, self.nbins[0]) * self.state_amp[0]
        self.s1 = np.linspace(-1, 1, self.nbins[1]) * self.state_amp[1]

        self.xv, self.yv = np.meshgrid(self.s0, self.s1)
        self.state_space = np.transpose(np.array([self.xv, self.yv]), (2, 1, 0))
        self.d_states = self.state_space.reshape((-1, 2))

        self.state_idx_mesh = np.meshgrid(range(self.nbins[0]), range(self.nbins[1]))
        self.state_idx_mesh = np.transpose(np.array(self.state_idx_mesh), (2, 1, 0)).reshape(-1, self.state_n)
        self.Qtable = np.random.normal(0, 1, (self.nbins[0], self.nbins[1], self.cntr_n))

    def get_qval(self, y):
        f0 = RectBivariateSpline(self.s0, self.s1, self.Qtable[:, :, 0])
        f1 = RectBivariateSpline(self.s0, self.s1, self.Qtable[:, :, 1])
        return np.array([f0.ev(y[0], y[1]), f1.ev(y[0], y[1])])

    def get_control(self, state):
        u_idx = np.argmax(self.get_qval(state))
        u_star = self.cntr_space[u_idx]
        q_star = self.get_qval(state)[u_idx]
        return u_idx, u_star, q_star

    def run_vi(self, state_update_func, cost_func, iterations=1):
        for _ in range(iterations):
            for s_idx in self.state_idx_mesh:
                state = self.state_space[s_idx[0], s_idx[1]]
                u_idx, u_star, q_star = self.get_control(state)
                state_new = state_update_func(state, u_star)
                cost = cost_func(state, u_star)
                self.discrete_update(s_idx, u_idx, -cost, q_star, state_new)

    def bellman(self, r0, q0_star, s1):
        q1_star = np.max(self.get_qval(s1))
        return (1 - self.alpha) * q0_star + self.alpha * (r0 + self.gamma * q1_star)

    def discrete_update(self, s_idx, u_idx, r0, q0_star, s1):
        self.Qtable[s_idx[0], s_idx[1], u_idx] = self.bellman(r0, q0_star, s1)

