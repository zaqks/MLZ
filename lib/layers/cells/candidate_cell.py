import numpy as np
from .base_cell import BaseCell
from ..activations.activation_layer import TanH

class CandidateCell(BaseCell):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)

    def backward(self, dc,i_t):    
      """
        dg :Grad of loss / g_t; (N, H)
      """
       
      dg = dc * i_t  
      return dg
