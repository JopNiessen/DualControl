"""
Soft Actor-Critic: main class
"""

# import local libraries
from src.RL.PolicyGradient.SoftActorCritic_v2.QFunction import *
from src.RL.PolicyGradient.SoftActorCritic_v2.ValueFunction import *
from src.RL.PolicyGradient.SoftActorCritic_v2.PolicyFunction import *

from src.utilities.ReplayBuffer import ReplayBuffer
from src.utilities.Tracker import Tracker

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom


class SoftActorCritic:
    """Soft Actor-Critic - main class"""
    def __init__(self, observation_size, control_size, key,
                    lr_q=1e-1, lr_pi=1e0,
                    alpha=.2,
                    batch_size=100, n_epochs=1,
                    buffer_size=1000):
        """
        :param observation_size: number of observables [int]
        :param control_size: number of control variables [int]
        :param key: PRNGKey
        :param lr_q: learning rate of Q-network [float]
        :param lr_pi: learning rate of policy-network [float]
        :param alpha: entropy regularization coefficient [float]
        :param batch_size: batch size [int]
        :param n_epochs: number of epochs per observation [int]
        :param buffer_size: buffer size [int]
        """
        self.n_obs = observation_size
        self.n_ctrl = control_size

        # Create actor and critic networks
        keys = jrandom.split(key, 5)
        self.SQF_1 = SoftQFunction(self.n_obs + self.n_ctrl, keys[1], eta=lr_q)
        self.SQF_2 = SoftQFunction(self.n_obs + self.n_ctrl, keys[2], eta=lr_q)
        self.PI = SoftPolicyFunction(self.n_obs, keys[3], alpha=alpha, eta=lr_pi)

        # Build replay buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size, observation_size, control_size, keys[4])
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Build tracker
        self.tracker = Tracker(['state0', 'state1', 'control', 'reward'])

        # Normalization factor
        self.max_cost = 4.1
    
    def cost_to_normalized_reward(self, x):
        """
        Cost to reward transformation incl. normalization
        :param x: cost [float]
        :return: normalized reward between (-1,1)
        """
        x = x/self.max_cost
        return 0 - min(x, 1)*1
    
    def q_func(self, state, control):
        """
        Calculate minimal q_value (accelerates learning)
        :param state: state
        :param control: control
        :return: minimal Q-value
        """
        q1 = self.SQF_1.predict(state, control)
        q2 = self.SQF_2.predict(state, control)
        Q = jax.lax.min(q1, q2)
        return Q
    
    def get_control(self, state, key):
        """
        Predict control
        :param state: state [array]
        :param key: PRNGKey
        :return: control [float], entropy [float]
        """
        return self.PI.get_control(state, key)

    def train_step(self, key, batch_size):
        """
        One iteration training
        :param key: PRNGKey
        :param batch_size: batch size [int]
        :return loss q1-network [float], loss q2-network [float], loss policy-network [float]
        """
        keys = jrandom.split(key, 4)

        # sample from buffer
        D = self.ReplayBuffer.sample_batch(batch_size)
        state = D[:, :self.n_obs]
        full_state = D[:, :self.n_obs+self.n_ctrl]
        reward = D[:, self.n_obs+self.n_ctrl:-self.n_obs]
        next_state = D[:, -self.n_obs:]

        # update critic
        loss_q1 = self.SQF_1.update(full_state, reward, next_state, self.PI, keys[1])
        loss_q2 = self.SQF_2.update(full_state, reward, next_state, self.PI, keys[2])

        # update actor
        loss_pi = self.PI.update(state, self.q_func, keys[3])

        return loss_q1, loss_q2, loss_pi
    
    def train(self, key, batch_size=100, n_epochs=5, show=False):
        """
        Train SAC
        :param key: PRNGKey
        :param batch_size: batch size [int]
        :param n_epochs: number of epochs [int]
        :param show: indicate if losses are printed during training [bool]
        """
        for epoch in range(n_epochs):
            key, _ = jrandom.split(key)
            loss_q1, loss_q2, loss_pi = self.train_step(key, batch_size=batch_size)
            if show:
                print(f'epoch={epoch} \t loss q1={loss_q1:.3f} \t loss q2={loss_q2:.3f} \t loss pi={loss_pi:.3f}')

    def update(self, state_transition, key, tracking=True, learning=True):
        """
        Update SAC
        :param state_transition: (state, control, reward, new state) [tuple]
        :param key: PRNGKey
        :param tracking: indicate if states are saved [boolean]
        :param learning: indicate if network parameters are updated [boolean]
        """
        state, control, cost, next_state = state_transition
        reward = self.cost_to_normalized_reward(cost)

        self.ReplayBuffer.store((state, control, reward, next_state))

        if learning:
            self.train(key, batch_size=self.batch_size, n_epochs=self.n_epochs)

        if tracking:
            self.tracker.add([state[0], state[1], control, reward])




