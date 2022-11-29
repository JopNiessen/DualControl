"""


"""

import jax.numpy as jnp


def quadratic(y_estimate, y_target):
    return (y_estimate - y_target)**2

def gaussian(y_estimate, y_target):
    sigma = .5
    norm = jnp.exp(.5*((y_estimate - y_target)/sigma)**2) / (sigma * jnp.sqrt(2*jnp.pi))
    return - jnp.log(norm) * quadratic(y_estimate, y_target)

