"""


"""

from src.RL.PolicyGradient.SoftActorCritic.SoftQFunction import *
from src.RL.PolicyGradient.SoftActorCritic.SoftValueFunction import *
from src.RL.PolicyGradient.SoftActorCritic.PolicyFunction import *

from src.utilities.Tracker import Tracker

import jax.numpy as jnp
import jax.random as jrandom

import random


class SoftActorCritic:
    def __init__(self, dim_q, dim_v, dim_pi, key):
        key_q, key_v, key_pi = jrandom.split(key, 3)
        self.SQF = SoftQFunction(dim_q, key_q)
        self.SVF = SoftValueFunction(dim_v, key_v)
        self.PI = SoftPolicyFunction(dim_pi, key_pi)
        self.buffer = list()
        self.tracker = Tracker(['state0', 'state1', 'control', 'cost', 'V_value', 'V_loss', 
                                    'Q_value', 'Q_loss', 'policy_angle', 'policy_force'])
        
        self.amplitude = 5
        self.n_dim = 4
        self.factor = int(self.n_dim/2)
        self.gamma = .95
        self.certainty = np.zeros((self.n_dim, self.n_dim, self.n_dim))
    
    def update(self, transition, key, tracking=True):
        state, control, reward, _ = transition
        k = min(len(self.buffer), 4)
        D = random.choices(self.buffer, k=k)
        D.append(transition)
        
        #TODO: certainty metric
        self.certainty = self.gamma * self.certainty
        state_ext = np.hstack([state, control])
        self.certainty[self.bind(state_ext)] += 1
        self.add_to_buffer(transition)

        v_value = self.SVF.predict(state, output_value=True)
        q_value = self.SQF.predict(state, control, output_value=True)
        
        v_loss = self.SVF.take_step(D, self.SQF.predict, self.PI.log_prob)
        q_loss = self.SQF.take_step(D, self.SVF.predict)
        self.PI.update(state, control, self.SQF.predict)

        if tracking:
            control_angle = jnp.arctan2(self.PI.params[0,0], self.PI.params[0,1])
            control_force = jnp.linalg.norm(self.PI.params)
            self.tracker.add([state[0], state[1], control, -reward, v_value, v_loss, q_value, q_loss,
                                    control_angle, control_force])
    
    def bind(self, state):
        state[state > self.amplitude] = self.amplitude
        state[state < -self.amplitude] = -self.amplitude
        return tuple((self.factor * state / (self.amplitude) - .5).astype(int) + self.factor)

    def get_control(self, state):
        return self.PI.get_control(state)
    
    def add_to_buffer(self, transition):
        #self.buffer = [transition]
        self.buffer.append(transition)

