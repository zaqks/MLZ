import numpy as np

from .base_layer import *



class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inpt):
        return np.reshape(inpt, self.output_shape)

    def backward(self, E_Y, a):
        return np.reshape(E_Y, self.input_shape)
