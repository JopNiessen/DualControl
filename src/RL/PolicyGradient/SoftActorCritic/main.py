"""
Soft Actor-Critic: main class
"""

# import local libraries
from src.RL.PolicyGradient.SoftActorCritic.SoftQFunction import *
from src.RL.PolicyGradient.SoftActorCritic.SoftValueFunction import *
from src.RL.PolicyGradient.SoftActorCritic.PolicyFunction import *

from src.utilities.ReplayBuffer import ReplayBuffer
from src.utilities.Tracker import Tracker

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom


class SoftActorCritic:
    """
    Soft Actor-Critic
    """
    def __init__(self, n_states, n_controls, key, n_hidden=32, buffer_size=100):
        """
        Initialize agent
        :param n_state: number of states [int]
        :param n_controls: number of controls [int]
        :param key: PRNGKey
        :param n_hidden: number of hidden states
        :param buffer_size: sice of replay buffer [int]
        """
        self.n_states = n_states
        self.n_ctrl = n_controls

        key_q1, key_q2, key_v, key_pi = jrandom.split(key, 4)
        self.SQF_1 = SoftQFunction((n_states + n_controls, n_hidden, 1), key_q1)
        self.SQF_2 = SoftQFunction((n_states + n_controls, n_hidden, 1), key_q2)
        self.SVF = SoftValueFunction((n_states, n_hidden, 1), key_v)
        self.PI = SoftPolicyFunction((n_states, n_controls), key_pi)
        
        # Build replay buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size, n_states, n_controls, key)
        self.batch_size = 20
        self.n_epochs = 1

        # Build state tracking
        self.tracker = Tracker(['state0', 'state1', 'control', 'reward', 'angle'])
        
        # Normalization
        self.max_cost = 26
    
    def q_value(self, state, control, output_value=True):
        """
        Calculate minimal q_value (accelerates learning)
        :param state: state
        :param control: control
        :param output_value: indicate if output should be [float] (True) or [DeviceArray] (False)
        :return: minimal Q-value
        """
        q1 = self.SQF_1.predict(state, control, output_value=output_value)
        q2 = self.SQF_2.predict(state, control, output_value=output_value)
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
        reward = cost_to_normalized_reward(cost) # scale reward between [0,1]

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
        self.SVF_bar = self.SVF
        for epoch in range(n_epochs):
            if epoch%100 == 0:
                self.SVF_bar = self.SVF
            key, _ = jrandom.split(key)
            loss_v, loss_q1, loss_q2, loss_pi = self.train_step(key, batch_size=batch_size)
            if show:
                weights = self.PI.model.mu_layer.weight[0]
                #angle = angle = jnp.arctan2(weights[0], weights[1])
                #print(f'angle={angle:.3f}')
                print(weights)
                #print(f'epoch={epoch} \t loss v={loss_v:.3f} \t loss q1={loss_q1:.3f} \t loss q2={loss_q2:.3f} \t loss pi={loss_pi:.3f}')

    def train_step(self, key, batch_size):
        """
        One epoch SAC training
        :param key: PRNGKey
        :param batch_size: batch size [int]
        """
        # sample from buffer
        D = self.ReplayBuffer.sample_batch(batch_size)

        # update SAC components
        loss_v = None
        #loss_v = self.SVF.take_step(D[:,:self.n_states],
        #                    self.q_value, self.get_control, key)
        loss_q1 = self.SQF_1.take_step(D[:, :self.n_states+self.n_ctrl],
                            D[:, self.n_states+self.n_ctrl:-self.n_states],
                            D[:, -self.n_states:],
                            self.SVF_bar.predict)
        loss_q2 = self.SQF_2.take_step(D[:, :self.n_states+self.n_ctrl],
                            D[:, self.n_states+self.n_ctrl:-self.n_states],
                            D[:, -self.n_states:],
                            self.SVF_bar.predict)
        loss_pi = self.PI.take_step(D[:,:self.n_states], self.q_value, key)

        return loss_v, loss_q1, loss_q2, loss_pi

    def get_control(self, state, key):
        """
        Infere control
        :param state: state
        :param key: PRNGKey
        :return: control
        """
        return self.PI.get_control(state, key)        



"""Functions"""
def cost_to_normalized_reward(x):
    x = x/4.1 #4.1 equals maximal cost per timestep
    return -min(x, 1)

