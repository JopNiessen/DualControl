"""



"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax

from jax import value_and_grad


class DQLearning:
    def __init__(self, DeepNet, n_obs):
        self.ctrl = jnp.array([-1, 1])

        self.n_ctrl = len(self.ctrl)
        self.n_obs = n_obs

        self.loss_func = quadratic_loss
        self.gamma = .9     # Discount factor

        self.network = DeepNet
        
        self.epsilon = .9   # Exploration rate
        self.epsilon_min = .1
        self.decay = .95

    def update(self, y0, y1, u, r0):
        u_idx = np.int(np.where(self.ctrl == u)[0])
        q1_hat = self.network.predict(y1)
        q0_hat = self.network.predict(y0, learning=True)
        q_target = q0_hat

        q_target = q_target.at[u_idx].set(r0 + self.gamma * jnp.max(q1_hat))
        loss = self.network.update(q0_hat, q_target)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        return loss
    
    def get_qval(self, y):
        q_hat = self.network.predict(y)
        return q_hat

    def optimal_control(self, y):
        q_hat = self.network.predict(y)
        u_idx = jnp.argmax(q_hat)
        return self.ctrl[u_idx]

    def get_control(self, y):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.ctrl)
        else:
            return self.optimal_control(y)




class NQLearning:
    def __init__(self, n_obs, key=jrandom.PRNGKey(0)):
        self.ctrl = jnp.array([-1, 1])

        self.n_ctrl = len(self.ctrl)
        self.n_obs = n_obs

        self.params = jrandom.normal(key, (self.n_ctrl, self.n_obs))

        self.loss_func = quadratic_loss
        self.gamma = .9     # Discount factor
        self.eta = 1e-6       # Learning rate
        
        self.epsilon = .9   # Exploration rate
        self.epsilon_min = .01
        self.decay = .95

    def loss(self, params, y0, y1, u_idx, r0):
        q_hat = jnp.dot(params, y0)[u_idx]
        q_target = r0 + self.gamma * np.max(jnp.dot(jax.lax.stop_gradient(params), y1))
        return self.loss_func(q_hat, q_target)

    def update(self, y0, y1, u, r0):
        u_idx = np.int(np.where(self.ctrl == u)[0])
        q_loss, grads = value_and_grad(self.loss)(self.params, y0, y1, u_idx, r0)
        self.params -= self.eta * grads
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        return q_loss

    def optimal_control(self, y):
        u_idx = jnp.argmax(jnp.dot(self.params, y))
        return self.ctrl[u_idx]

    def get_control(self, y):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.ctrl)
        else:
            return self.optimal_control(y)


"""Loss functions"""

def quadratic_loss(target, estimate):
    return(target - estimate)**2

