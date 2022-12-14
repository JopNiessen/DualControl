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
        self.batch_size = 5

        # Build state tracking
        self.tracker = Tracker(['state0', 'state1', 'control', 'cost'])
        
        # Normalization
        self.max_cost = 50
    
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
        state, control, reward, new_state = state_transition
        #reward = max(reward, -self.max_cost) / self.max_cost + 1 # scale reward between [0,1]

        self.ReplayBuffer.store((state, control, reward, new_state))
        self.train(key, batch_size=self.batch_size)

        if tracking:
            self.tracker.add([state[0], state[1], control, -reward])
    
    def train(self, key, batch_size=5, n_epochs=1, show=False):
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

    def train_step(self, key, batch_size=10):
        """
        One epoch SAC training
        :param key: PRNGKey
        :param batch_size: batch size [int]
        """
        # sample from buffer
        D = self.ReplayBuffer.sample_batch(batch_size)

        # update SAC components
        loss_v = self.SVF.take_step(D[:,:self.n_states], D[:,self.n_states:self.n_states+self.n_ctrl], self.q_value, self.PI.log_prob, self.get_control, key)
        loss_q1 = self.SQF_1.take_step(D[:, :self.n_states+self.n_ctrl],
                            D[:, self.n_states+self.n_ctrl:-self.n_states],
                            D[:, -self.n_states:],
                            self.SVF.predict)
        loss_q2 = self.SQF_1.take_step(D[:, :self.n_states+self.n_ctrl],
                            D[:, self.n_states+self.n_ctrl:-self.n_states],
                            D[:, -self.n_states:],
                            self.SVF.predict)
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


"""
    def bind(self, state):
        state[state > self.amplitude] = self.amplitude
        state[state < -self.amplitude] = -self.amplitude
        return tuple((self.factor * state / (self.amplitude) - .5).astype(int) + self.factor)

"""

