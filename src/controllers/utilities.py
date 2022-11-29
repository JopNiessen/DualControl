"""

"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


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

    if training_wheels:
        system.boundary = 5
        tw_increase = 0

    for ti, t in enumerate(time_horizon):

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

        if training_wheels:
            if done:
                system.reset(x0)
            system.boundary += tw_increase

    return time_horizon, X, U, C, L, controller


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

