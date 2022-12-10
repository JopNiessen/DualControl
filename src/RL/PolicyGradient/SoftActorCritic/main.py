"""


"""

from src.RL.PolicyGradient.SoftActorCritic.SoftQFunction import *
from src.RL.PolicyGradient.SoftActorCritic.SoftValueFunction import *
from src.RL.PolicyGradient.SoftActorCritic.PolicyFunction import *

from src.utilities.Tracker import Tracker

import jax.numpy as jnp
import jax.random as jrandom


class SoftActorCritic:
    def __init__(self, dim_q, dim_v, dim_pi, key):
        key_q, key_v, key_pi = jrandom.split(key, 3)
        self.SQF = SoftQFunction(dim_q, key_q)
        self.SVF = SoftValueFunction(dim_v, key_v)
        self.PI = SoftPolicyFunction(dim_pi, key_pi)
        self.buffer = list()
        self.tracker = Tracker(['state0', 'state1', 'control', 'cost', 'V_value', 'V_loss', 
                                    'Q_value', 'Q_loss', 'policy_angle', 'policy_force'])
    
    def update(self, state, control, cost, tracking=True):
        v_value = self.SVF.predict(state, output_value=True)
        q_value = self.SQF.predict(state, control, output_value=True)
        
        v_loss = self.SVF.take_step(self.buffer, self.SQF.predict, self.PI.log_prob)
        q_loss = self.SQF.take_step(self.buffer, self.SVF.predict)
        self.PI.update(state, control, self.SQF.predict)

        if tracking:
            control_angle = jnp.arctan2(self.PI.params[0,0], self.PI.params[0,1])
            control_force = jnp.linalg.norm(self.PI.params)
            self.tracker.add([state[0], state[1], control, cost, v_value, v_loss, q_value, q_loss,
                                    control_angle, control_force])
    
    def get_control(self, state):
        return self.PI.get_control(state)
    
    def add_to_buffer(self, transition):
        self.buffer.append(transition)

