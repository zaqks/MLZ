import numpy as np
from .base_cell import BaseCell
from ..activations.activation_layer import TanH

class Cell(BaseCell):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)

    def forward(self, f_t, i_t, g_t, c_prev):
        c_t = f_t * c_prev + i_t * g_t
        return c_t


    def backward(self, dc_next, dht, o_t, c_t):
   
        dc = dht * o_t * TanH().activation_prime(c_t) + dc_next

        return dc    

class HiddenState:
    def __init__(self, hidden_size):
        self.h = np.zeros(hidden_size)  

    def update(self, o_t, c_t):
        self.h = o_t * TanH().activation(c_t)  # Update hidden state
        return self.h