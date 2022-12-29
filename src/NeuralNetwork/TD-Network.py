"""

"""

from NeuralNetwork import NeuralNet

class TD_NeuralNet(NeuralNet):
    def __init__(self, key, dimension, activation, eta=1e-2):
        super().__init__(key, dimension, activation, eta=eta)
        
        self.gamma = .9
    
    def loss(self, x0, x1, reward):
        V0 = self.predict(x0)
        V1 = self.predict(x1)
        return reward + self.gamma * V1 - V0
    




