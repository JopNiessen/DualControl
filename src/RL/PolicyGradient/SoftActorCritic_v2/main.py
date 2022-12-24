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
    """
    Soft Actor-Critic
    """
    def __init__(self, n_obs, n_controls, key, n_hidden=32, buffer_size=100):
        """
        Initialize agent
        :param n_state: number of states [int]
        :param n_controls: number of controls [int]
        :param key: PRNGKey
        :param n_hidden: number of hidden states
        :param buffer_size: sice of replay buffer [int]
        """
        self.n_obs = n_obs
        self.n_ctrl = n_controls

        key_q1, key_q2, key_v, key_pi = jrandom.split(key, 4)
        self.SQF_1 = SoftQFunction(n_obs + n_controls, key_q1)
        self.SQF_2 = SoftQFunction(n_obs + n_controls, key_q2)
        self.SVF = SoftValueFunction(n_obs, key_v)
        self.PI = SoftPolicyFunction(n_obs, key_pi)
        
        # Build replay buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size, n_obs, n_controls, key)
        self.batch_size = 100
        self.n_epochs = 1

        # Build state tracking
        self.tracker = Tracker(['state0', 'state1', 'control', 'reward'])
        
        # Normalization
        self.max_cost = 4.1
    
    """Functions"""
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

    def update(self, state_transition, key, tracking=True):
        """
        Update SAC
        :param state_transition: (state, control, reward, new state) [tuple]
        :param key: PRNGKey
        :param tracking: boolean indicates if states are saved [boolean]
        """
        
        # reward normalization
        state, control, cost, new_state = state_transition
        reward = self.cost_to_normalized_reward(cost) # scale reward between [0,1]

        self.ReplayBuffer.store((state, control, reward, new_state))
        self.train(key, batch_size=self.batch_size, n_epochs=self.n_epochs)

        if tracking:
            weights = self.PI.model.mu_layer.weight[0]
            angle = jnp.arctan2(weights[0], weights[1])
            self.tracker.add([state[0], state[1], control, reward, angle])
    
    def train(self, key, batch_size=5, n_epochs=2, show=False):
        """
        train SAC components
        :param key: PRNGKey
        :param batch_size: batch size [int]
        :param n_epochs: number of epochs [int]
        :param show: indicate if loss is shown [boolean]
        """
        for epoch in range(n_epochs):
            key, _ = jrandom.split(key)
            loss_v, loss_q1, loss_q2, loss_pi = self.train_step(key, batch_size=batch_size)
            if show:
                print(f'epoch={epoch} \t loss v={loss_v:.3f} \t loss q1={loss_q1:.3f} \t loss q2={loss_q2:.3f} \t loss pi={loss_pi:.3f}')

    def train_step(self, key, batch_size):
        """
        One epoch SAC training
        :param key: PRNGKey
        :param batch_size: batch size [int]
        """
        keys = jrandom.split(key, 4)
        # sample from buffer
        D = self.ReplayBuffer.sample_batch(batch_size)

        # update SAC components
        state = D[:, :self.n_obs]
        full_state = D[:, :self.n_obs+self.n_ctrl]
        reward = D[:, self.n_obs+self.n_ctrl:-self.n_obs]
        next_state = D[:, -self.n_obs:]

        loss_v = self.SVF.update(state, self.q_func, self.PI, keys[0])
        loss_q1 = self.SQF_1.update(full_state, reward,
                                    next_state, self.PI, keys[1])
        loss_q2 = self.SQF_2.update(full_state, reward,
                                    next_state, self.PI, keys[2])
        loss_pi = self.PI.update(state, self.q_func, keys[3])
        return loss_v, loss_q1, loss_q2, loss_pi

    def get_control(self, state, key):
        """
        Infere control
        :param state: state
        :param key: PRNGKey
        :return: control
        """
        return self.PI.get_control(state, key)        





