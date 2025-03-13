from .base_gate import gate
from ..activations.activation_layer import TanH, Sigmoid
import numpy as np

class Input_gate(gate):
    def __init__(self, input_size, hidden_size):
        self.Wxi = np.random.randn(input_size, hidden_size)  # Shape: (I, H)
        self.Whi = np.random.randn(hidden_size, hidden_size)  # Shape: (H, H)
        self.bi = np.random.randn(hidden_size)  # Bias for the gate; Shape: (H,)


    def activation(self, x):
        """ Sigmoid activation function for input gate. """
        return 1 / (1 + np.exp(-x))

    def forward(self, x_t, h_prev):
        """
        x_t : (N, D)
        h_prev : (N, H)
        """
        i_t = Sigmoid().activation(np.dot(x_t, self.Wxi) + np.dot(h_prev, self.Whi) + self.bi)
        return i_t #Shape: (N, H)

    def backward(self, dL_dc, dL_dht, o_t, x_t, h_prev,c_t):
        """

        Parameters:
        dL_dc : Grad L /grad c_t; Shape: (N, H)
        dL_dht : Grad L /grad h_t; Shape: (N, H)
        """        
        dL_di = dL_dc * (TanH().activation(np.dot(x_t, self.Wxi) + np.dot(h_prev, self.Whi))) + dL_dht * o_t * TanH().activation_prime(c_t) 
        return dL_di #Shape:(N,H)
