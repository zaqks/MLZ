from .base_layer import Layer
import numpy as np


"""
R x C for matrices 
"""


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)  # input X out
        # random mat 1 X output_size
        self.bias = np.random.randn(output_size, 1)

    def forward(self, inpt):
        self.input = inpt
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, E_Y, a):
        E_X = np.dot(self.weights.T, E_Y)

        E_W = np.dot(E_Y, self.input.T)
        
        # we update the 2 params
        self.weights -= a*E_W
        self.bias -= a*E_Y

        return E_X
