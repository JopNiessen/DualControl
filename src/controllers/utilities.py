"""

"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


""" Methods for Soft Actor-Critic """

def run_SAC(key, system, controller, T):
    key, subkey = jrandom.split(key)
    time_horizon = np.arange(0, T, system.dt)

    for _ in time_horizon:
        s0_estimate = system.observe(key)
        u, _ = controller.get_control(s0_estimate)
        _, cost, done = system.update(key, u, info=True)
        s1_estimate = system.observe(subkey)
        controller.add_to_buffer((s0_estimate, u, -cost, s1_estimate))

        controller.update(s0_estimate, u)
        
        # step
        key, subkey = jrandom.split(key)

        if done:
            x0 = jrandom.normal(key, (2,))*2
            system.reset(x0)
    
    return system, controller, time_horizon



""" Methods for Deep Q-learning and REINFORCE """

def run_controlled_environment(key, controller, system, T, 
                        learning=True, training_wheels=False,
                        method='ValueFunction'):
    x0 = jnp.array([0, 0])

    time_horizon = np.arange(0, T, system.dt)

    n_steps = len(time_horizon)
    n_obs = system.dim

    X = np.zeros((n_steps, n_obs))
    U = np.zeros(n_steps)
    C = np.zeros(n_steps)
    L = np.zeros(n_steps)
    P = np.zeros(n_steps)
    M = np.zeros(n_steps)

    if training_wheels:
        system.boundary = 5
        tw_increase = 0

    for ti, t in enumerate(time_horizon):
        P[ti] = np.arctan2(controller.params[0, 0], controller.params[0, 1])

        if method == 'ValueFunction':
            controller, system, state, u_star, cost, loss, done = ValueFunction_step(key, system, controller, learning)
        elif method == 'PolicyGradient':
            controller, system, state, u_star, cost, loss, done = PolicyGradient_step(key, system, controller, learning)
        else:
            raise TypeError('Method not supported.')
        key, subkey = jrandom.split(key)

        # save state
        X[ti] = state
        U[ti] = u_star
        C[ti] = cost
        L[ti] = loss
        M[ti] = np.linalg.norm(controller.params[0])

        if training_wheels:
            if done:
                x0 = jrandom.normal(key, (2, ))*2
                system.reset(x0)
            system.boundary += tw_increase

    return time_horizon, X, U, C, L, P, M, controller


def PolicyGradient_step(key, system, controller, learning):
    key, subkey = jrandom.split(key)
    y0 = system.observe(subkey)
    #controller.run_simulation(system)
    u_star = controller.get_control(y0)

    state, cost, done = system.update(key, u_star, info=True)
    if learning:
        controller.update(y0, u_star, -cost, done)

    return controller, system, state, u_star, cost, None, done


def ValueFunction_step(key, system, controller, learning):
    key, subkey = jrandom.split(key)
    y0 = system.observe(subkey)
    u_star = controller.get_control(y0)

    # state update
    state, cost, done = system.update(key, u_star, info=True)
    y1 = system.observe(subkey)

    # learning step
    if learning:
        loss = controller.update(y0, y1, u_star, -cost)
    else:
        loss = None
    return controller, system, state, u_star, cost, loss, done

    # key, subkey = jrandom.split(key)
    # y0 = system.observe(subkey)
    # u_star = controller.get_control(y0)

    # # state update
    # state, cost = system.update(key, u_star, info=True)
    # y1 = system.observe(subkey)

    # # learning step
    # if learning:
    #     loss = controller.update(y0, y1, u_star, -cost)
    # else:
    #     loss = None
    
    # return controller, system, state, u_star, cost


"""
UPDATE
"""

# def run_controlled_environment(key, controller, system, T, 
#                         learning=True, training_wheels=False):
#     x0 = jnp.array([0, 0])

#     time_horizon = np.arange(0, T, system.dt)

#     n_steps = len(time_horizon)
#     n_obs = system.dim

#     X = np.zeros((n_steps, n_obs))
#     U = np.zeros(n_steps)
#     C = np.zeros(n_steps)
#     L = np.zeros(n_steps)

#     if training_wheels:
#         tw_width = 5
#         tw_increase = 0

#     for ti, t in enumerate(time_horizon):
#         key, subkey = jrandom.split(key)
#         y0 = system.observe(subkey)
#         u_star = controller.get_control(y0)

#         # state update
#         state, cost = system.update(key, u_star, info=True)
#         y1 = system.observe(subkey)

#         # learning step
#         if learning:
#             L[ti] = controller.update(y0, y1, u_star, -cost)

#         # save state
#         X[ti] = state
#         U[ti] = u_star
#         C[ti] = cost

#         if training_wheels:
#             if abs(state[0]) >= tw_width:
#                 system.reset(x0)
#             training_wheels += tw_increase

#     return time_horizon, X, U, C, L, controller

