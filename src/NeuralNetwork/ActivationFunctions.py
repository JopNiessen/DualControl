"""

"""

import jax.numpy as jnp
import jax


# Sigmoid
def sigmoid(x):
  return 1 / (1 + jnp.exp(-x))

def sigmoid_deriv(dA, x):
  return dA * sigmoid(x) * (1 - sigmoid(x))

# ReLU
def relu(x):
    return jax.nn.relu(x)

def relu_deriv(dA, x):
    dx = jnp.array(dA, copy=True)
    dx = dx.at[x <= 0].set(0)
    #dx[x <= 0] = 0
    return dx

# Linear
def linear(x):
    return x

def linear_deriv(dA, x):
    return jnp.array(dA, copy=True)


# Hyperbolic tanges
def tanh(x):
    return jnp.tanh(x)

def tanh_deriv(dA, x):
    return dA  / (jnp.cosh(x)**2)

