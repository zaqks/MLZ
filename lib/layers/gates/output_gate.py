from .base_gate import gate
from ..activations.activation_layer import TanH, Sigmoid
import numpy as np

class Output_gate(gate):
    def __init__(self, input_size, hidden_size):
        self.Wxo = np.random.randn(input_size, hidden_size)  # Shape: (I, H)
        self.Who = np.random.randn(hidden_size, hidden_size)  # Shape: (H, H)
        self.bo = np.random.randn(hidden_size)  # Bias for the gate; Shape: (H,)


    def forward(self, x_t, h_prev):
        """ 
        Parameters:
        x_t :(N, D)
        h_prev :(N, H)
        """
        
        o_t = self.Sigmoid(np.dot(x_t, self.Wxo) + np.dot(h_prev, self.Who) + self.bo)
        return o_t #Shape: (N, H)

    def backward(self, dL_dht, c_t):
        """
        Params:
        dL_dht :Grad L /grad h_t; Shape: (N, H)
        """
        
        dL_do = dL_dht * tanh(c_t) #type:ignore
        return dL_do #Shape:(N,H)
