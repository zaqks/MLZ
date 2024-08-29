import numpy as np
from scipy import signal

from .base_layer import Layer


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # depth is kernels count <=> output size

        input_depth, input_height, input_width = input_shape

        self.depth = depth

        self.input_shape = input_shape
        self.input_depth = input_depth

        self.output_shape = (depth, input_height -
                             kernel_size+1, input_width-kernel_size+1)

        self.kernels_shape = (
            depth, input_depth, kernel_size, kernel_size
        )

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)


    def forward(self, inpt):        
        self.input = inpt
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )

        return self.output

    def backward(self, E_Y, a=0.1):
        E_K = np.zeros(self.kernels_shape)
        E_X = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                E_K[i, j] = signal.correlate2d(self.input[j], E_Y[i], "valid")
                E_X [j] += signal.convolve2d(E_Y[i], self.kernels[i, j], "full")

        

        self.kernels -= a*E_K
        self.biases -= a*E_Y

        return E_X