from .base_layer import Layer
from ..optimizers.optimizers import GradientDescent

import numpy as np


"""
R x C for matrices 
"""


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # self.weights = np.random.randn(output_size, input_size)  # input X out
        # random mat 1 X output_size
        # self.biases = np.random.randn(output_size, 1)

        super().__init__(w=np.random.randn(output_size, input_size),
                         b=np.random.randn(output_size, 1))

    def forward(self, inpt):
        self.input = inpt
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, E_Y, a):
        E_X = np.dot(self.weights.T, E_Y)

        E_W = np.dot(E_Y, self.input.T)

        # we update the 2 params          
        self.weights -= self.optimizer.formula(a, E_W, 0)
        self.biases -= self.optimizer.formula(a, E_Y, 1)
        
        return E_X
