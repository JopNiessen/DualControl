"""

"""

from src.NeuralNetwork.ActivationFunctions import *

import jax.numpy as jnp
import jax.random as jrandom


class NeuralNet:
    def __init__(self, key, dimension, activation, eta=1e-4):
        """
        Neural network
        :param key: random key
        :param dimension: network dimensions (list)
        :param activation: activation functions (list)
        :param eta: learning rate
        """
        self.dim = dimension
        self.input_shape = (self.dim[0], 1)
        self.output_shape = (self.dim[-1], 1)

        self.params = {}
        self.layers = activation

        self.act_fun = {'sigmoid': sigmoid, 'relu': relu, 'linear': linear, 'tanh': tanh}
        self.dact_fun = {'sigmoid': sigmoid_deriv, 'relu': relu_deriv, 'linear': linear_deriv, 'tanh': tanh_deriv}

        self.init_network(key, activation)
        self.memory = {}

        self.eta = eta      # learning rate
    
    def init_network(self, key, activation):
        """
        Initialize network
        :param key: random key
        :param activation: activation functions (list)
        """
        self.params = {}
        for idx, act in enumerate(activation):
            n_input = self.dim[idx]
            n_output = self.dim[idx+1]
            self.params['W' + str(idx)] = jrandom.normal(key, (n_output, n_input))
            self.params['b' + str(idx)] = jnp.zeros((n_output, 1))
    
    def forward_step(self, a, W, b, activation='relu'):
        """
        One step forward pass
        :param a: activated signal previous layer (array)
        :param W: weights current layer (matrix)
        :param b: bias current layer (array)
        :param activation: activation function current layer
        :return: [estimated output signal, estimated hidden state]
        """
        h_hat = jnp.dot(W, a) + b
        a_hat = self.act_fun[activation](h_hat)
        return a_hat, h_hat
    
    def forward_propagation(self, input, learning=True):
        """
        Full network forward pass
        :param input: network input
        :param learning: (boolean) verify nodes should store information for learning
        :return: network output (array)
        """
        input = input.reshape(self.input_shape)
        a_curr = input
        for idx, activation in enumerate(self.layers):
            a_prev = a_curr
            W = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            a_curr, h_curr = self.forward_step(a_prev, W, b, activation)
            
            if learning:
                self.memory['a' + str(idx)] = a_prev
                self.memory['h' + str(idx)] = h_curr
        return a_curr
    
    def backward_step(self, dA_curr, W_curr, h_curr, A_prev, activation):
        """
        One step backward pass
        :param dA_curr: gradient of activated output current layer (array)
        :param W_curr: weights current layer (matrix)
        :param h_curr: hiden state current layer (array)
        :param A_prev: activated output previous layer (array)
        :param activation: activation function current layer
        
        :return dA_prev: gradient activated output previous layer (array)
        :return dW: gradient of the weights (matrix)
        :return db: gradient of the bias (array)
        """
        m = A_prev.shape[1]

        deriv_act_fun = self.dact_fun[activation]
        
        dh_curr = deriv_act_fun(dA_curr, h_curr)
        dW = jnp.dot(dh_curr, A_prev.T)/m
        db = jnp.sum(dh_curr, axis=1, keepdims=True)/m
        dA_prev = jnp.dot(W_curr.T, dh_curr)
        return dA_prev, dW, db
    
    def backpropagation(self, loss):
        """
        Full network backpropagation
        :param loss: network loss
        :return: gradients of network weights and biases
        """
        grads = {}

        dA_prev = loss
        #dA_prev = - (jnp.divide(y_target, y_hat) - jnp.divide(1 - y_hat, 1 - y_target)) #TODO
        for idx, activation in reversed(list(enumerate(self.layers))):
            dA_curr = dA_prev

            W_curr = self.params['W' + str(idx)]
            A_prev = self.memory['a' + str(idx)]
            h_curr = self.memory['h' + str(idx)]
            dA_prev, dW_curr, db_curr = self.backward_step(dA_curr, W_curr, h_curr, A_prev, activation)

            grads['dW' + str(idx)] = dW_curr
            grads['db' + str(idx)] = db_curr
        return grads

    def update_params(self, grads):
        """
        Update network weights W = W - eta grad(W)
        :param grads: gradients of network weights and biases
        """
        for idx in range(len(self.dim)-1):
            self.params['W' + str(idx)] -= self.eta * grads['dW' + str(idx)]
            #self.params['b' + str(idx)] -= self.eta * grads['db' + str(idx)]
    
    def loss(self, y_hat, y_target):
        """
        loss function
        :param y_hat: network output
        :param y_target: target output
        :return: quadratic loss
        """
        return (y_hat - y_target)**2

    def train_step(self, input, y_target):
        """
        One observation network training
        :param input: network input (array)
        :param y_target: target output
        :return: loss
        """
        y_hat = self.forward_propagation(input)
        loss = self.update(y_hat, y_target)
        return loss
    
    def update(self, y_hat, y_target):
        """
        Update network parameters
        :param y_hat: network output (array)
        :param y_target: target output (array)
        :return: sum of the loss (int)
        """
        loss = self.loss(y_hat, y_target)
        grads = self.backpropagation(loss)
        self.update_params(grads)
        return jnp.sum(loss)
    
    def train_batch(self, input, target):
        """
        Train network on batch of observations
        :param input: inputs (matrix)
        :param target: target outputs (matrix)
        :return: average loss
        """
        batch_size = len(input)
        sum_loss = 0
        for x, y_target in zip(input, target):
            sum_loss += self.train_step(x, y_target)
        return sum_loss / batch_size
    
    def predict(self, input, learning=False):
        """
        Make prediction
        :param input: network input
        :param learning: boolean indicating if learning information should be stored
        :return: prediction
        """
        y_hat = self.forward_propagation(input, learning=learning)
        return y_hat
    
    def predict_batch(self, input):
        """
        Make prediction for batch of network inputs
        :param input: network inputs (matrix)
        :return: predictions (matrix)
        """
        batch_size = len(input)
        Y_hat = jnp.zeros((batch_size, self.dim[-1]))
        for i, x in enumerate(input):
            y_hat = self.predict(x)
            Y_hat = Y_hat.at[i].set(y_hat[0])
        return Y_hat




""" OLD SCRIPTS """

class NN2:
    def __init__(self, key, dimension, act_fun, deriv_act_fun, eta=1e-6):
        self.dim = dimension

        self.W0 = jrandom.normal(key, (dimension[0], dimension[1]))
        self.W1 = jrandom.normal(key, (dimension[1], dimension[2]))

        self.eta = eta
        self.act_fun = act_fun
        self.deriv_act_fun = deriv_act_fun

    def predict_batch(self, input):
        Y = jnp.zeros((len(input), self.dim[-1]))
        for i, x in enumerate(input):
            y = self.predict(x)
            Y[i] = y
        return Y

    def predict(self, input):
        _, y_hat = self.forward_pass(input)
        return y_hat[0]

    def train(self, input, target):
        for x, y in zip(input, target):
            self.train_step(x, y)

    def train_step(self, input, target):
        h_hat, y_hat = self.forward_pass(input)
        grad_W0, grad_W1 = self.compute_gradient(input, h_hat, y_hat, target)
        self.weight_update(grad_W0, grad_W1)
    
    def forward_pass(self, input):
        h_hat = jnp.dot(self.W0, input)
        a_hat = self.act_fun(h_hat)
        y_hat = jnp.dot(self.W1, a_hat)
        return h_hat, y_hat

    def compute_gradient(self, input, h_hat, y_hat, target):
        error = target - y_hat
        grad_W1 = jnp.dot(h_hat.reshape((self.dim[-2], 1)),
                            error * self.deriv_act_fun(y_hat.reshape(1, self.dim[-1])))
        grad_W0 = jnp.dot(input.reshape((self.dim[0], 1)), 
                            (jnp.dot(error * self.deriv_act_fun(y_hat), self.W1.T) * self.deriv_act_fun(jnp.dot(input, self.W0))).reshape((1, self.dim[1])))
        return grad_W0, grad_W1

    def weight_update(self, grad_W0, grad_W1):
        self.W0 += self.eta * grad_W0
        self.W1 += self.eta * grad_W1




