from .base_gate import gate
from ..activations.activation_layer import TanH, Sigmoid
import numpy as np
# hidden_size :number of hidden units
#input_size: number of inut features
class forget_gate(gate):
     def __init__(self, input_size, hidden_size):
        
        self.Wxf = np.random.randn(input_size, hidden_size)  # Shape: (I, H)
        self.Whf = np.random.randn(hidden_size, hidden_size)  # Shape: (H, H)
        self.bf = np.random.randn(hidden_size)  # Bias for the gate; Shape: (H,)

  
     def forward(self, x_t, h_prev):  
         f_t = Sigmoid().activation(np.dot(x_t, self.Wxf) + np.dot(h_prev, self.Whf) + self.bf)
         #should i use multiply?
         return f_t #Shape: (N, H)


     def backward(self, dL_dht, dL_dc_next, i_t, c_prev, c_t):
        """
        Param:
        dL_dht :Grad L / grad h_t;(N, H)
        dL_dc_next :Grad L/ grad c_t; Shape: (N, H) (from next step)
        c_t,c_prev :(N, H)
        """
        dL_df = dL_dht * TanH().activation(c_prev) + dL_dc_next * TanH().activation_prime(c_t) # type: ignore
        return dL_df #Shape: (N, H)