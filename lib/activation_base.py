from .base_layer import Layer
import numpy as np


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inpt):
        self.input = inpt
        return self.activation(self.input)

    def backward(self, E_Y, a):
        # hadamar product
        return np.multiply(E_Y, self.activation_prime(self.input))
