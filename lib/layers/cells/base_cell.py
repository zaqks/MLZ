import numpy as np
from ..activations.activation_layer import TanH

class BaseCell:
    def __init__(self, input_size, hidden_size):
        self.Wxc = np.random.randn(input_size, hidden_size)  
        self.Whc = np.random.randn(hidden_size, hidden_size)  
        self.bc = np.random.randn(hidden_size)                

    def forward(self, x_t, h_prev):
        '''
        candidate cell output
        '''
        g_t = TanH().activation(np.dot(x_t, self.Wxc) + np.dot(h_prev, self.Whc) + self.bc)
        return g_t    