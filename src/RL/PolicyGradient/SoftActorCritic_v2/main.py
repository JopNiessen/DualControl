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
    def __init__(self, observation_size, control_size, key,
                    lr_v=1e-1, lr_q=1e-1, lr_pi=1e-1,
                    batch_size=100, n_epochs=1):
        self.n_obs = observation_size
        self.n_ctrl = control_size

        keys = jrandom.split(key, 5)
        #self.SVF = SoftValueFunction(self.n_obs, keys[0], eta=lr_v)
        self.SQF_1 = SoftQFunction(self.n_obs + self.n_ctrl, keys[1], eta=lr_q)
        self.SQF_2 = SoftQFunction(self.n_obs + self.n_ctrl, keys[2], eta=lr_q)
        self.PI = SoftPolicyFunction(self.n_obs, keys[3], eta=lr_pi)

        # Build replay buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size, observation_size, control_size, keys[4])
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Build tracker
        self.tracker = Tracker(['state0', 'state1', 'control', 'reward'])

        # Normalization
        self.max_cost = 4.1
    
    def cost_to_normalized_reward(self, x):
        """
        Cost to reward transformation incl. normalization
        :param x: cost [float]
        :return: normalized reward between (-1,1)
        """
        x = x/self.max_cost
        return 1 - min(x, 1)*2
    
    def q_func(self, state, control):
        """
        Calculate minimal q_value (accelerates learning)
        :param state: state
        :param control: control
        :param output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :return: minimal Q-value
        """
        q1 = self.SQF_1.predict(state, control)
        q2 = self.SQF_2.predict(state, control)
        Q = jax.lax.min(q1, q2)
        return Q
    
    def get_control(self, state, key):
        return self.PI.get_control(state, key)

    def train_step(self, key, batch_size):
        keys = jrandom.split(key, 4)

        # sample from buffer
        D = self.ReplayBuffer.sample_batch(batch_size)

        # update SAC components
        state = D[:, :self.n_obs]
        full_state = D[:, :self.n_obs+self.n_ctrl]
        reward = D[:, self.n_obs+self.n_ctrl:-self.n_obs]
        next_state = D[:, -self.n_obs:]

        # Update
        loss_v = None #self.SVF.update(state, self.q_func, PI, keys[0])
        loss_q1 = self.SQF_1.update(full_state, reward, next_state, self.PI, keys[1])
        loss_q2 = self.SQF_2.update(full_state, reward, next_state, self.PI, keys[2])
        loss_pi = self.PI.update(state, self.q_func, keys[3])

        return loss_v, loss_q1, loss_q2, loss_pi
    
    def train(self, key, batch_size=100, n_epochs=5, show=False):
        for epoch in range(n_epochs):
            key, _ = jrandom.split(key)
            loss_v, loss_q1, loss_q2, loss_pi = self.train_step(key, batch_size=batch_size)
            if show:
                print(f'epoch={epoch} \t loss v={loss_v:.3f} \t loss q1={loss_q1:.3f} \t loss q2={loss_q2:.3f} \t loss pi={loss_pi:.3f}')

    def update(self, state_transition, key, tracking=True):
        """
        Update SAC
        :param state_transition: (state, control, reward, new state) [tuple]
        :param key: PRNGKey
        :param tracking: boolean indicates if states are saved [boolean]
        """
        state, control, cost, next_state = state_transition
        reward = self.cost_to_normalized_reward(cost)

        self.ReplayBuffer.store((state, control, reward, next_state))
        self.train(key, batch_size=self.batch_size, n_epochs=self.n_epochs)

        if tracking:
            self.tracker.add([state[0], state[1], control, reward])




