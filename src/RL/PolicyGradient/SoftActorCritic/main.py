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

        # Build state tracking
        self.tracker = Tracker(['state0', 'state1', 'control', 'cost', 'V_value', 'V_loss', 
                                    'Q_value', 'Q_loss', 'policy_angle', 'policy_force'])
        # Normalization
        self.max_cost = 50
    
    def q_value(self, state, control, output_value=True):
        q1 = self.SQF_1.predict(state, control, output_value=output_value)
        q2 = self.SQF_2.predict(state, control, output_value=output_value)
        Q = jax.lax.min(q1, q2)
        return Q

    def update(self, state_transition, key, tracking=True):
        state, control, reward, new_state = state_transition
        #reward = max(reward, -self.max_cost) / self.max_cost +1 # scale reward between [0,1]
        self.ReplayBuffer.store((state, control, reward, new_state))
        D = self.ReplayBuffer.sample_batch(10)

        value_v = self.SVF.predict(state, output_value=True)
        value_q1 = self.SQF_1.predict(state, control, output_value=True)
        value_q2 = self.SQF_2.predict(state, control, output_value=True)
        
        loss_v = self.SVF.take_step(D, self.q_value, self.PI.log_prob, self.get_control, key)
        loss_q1 = self.SQF_1.take_step(D, self.SVF.predict)
        loss_q2 = self.SQF_2.take_step(D, self.SVF.predict)
        loss_pi = self.PI.take_step(state, self.q_value, key)

        if tracking:
            control_angle = None #jnp.arctan2(self.PI.params[0,0], self.PI.params[0,1])
            control_force = None #jnp.linalg.norm(self.PI.params)
            self.tracker.add([state[0], state[1], control, -reward, value_v, loss_v, value_q1, loss_q1,
                                    control_angle, control_force])
    
    def bind(self, state):
        state[state > self.amplitude] = self.amplitude
        state[state < -self.amplitude] = -self.amplitude
        return tuple((self.factor * state / (self.amplitude) - .5).astype(int) + self.factor)

    def get_control(self, state, key):
        return self.PI.get_control(state, key)
