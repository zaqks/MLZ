from .base_layer import Layer
import numpy as np


"""
Row and Columns are inverted in numpy
its not R X C
ITS C X R
"""

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # input X out       
        self.bias = np.random.randn(output_size, 1) # random mat 1 X output_size