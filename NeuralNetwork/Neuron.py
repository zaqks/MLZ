from .Funcs import Funcs
from random import random

INIT_VAL = None


class Neuron:
    def __init__(self, wn, activation=None):
        self.weights = [INIT_VAL for _ in range(wn)]
        self.bias = INIT_VAL

        if INIT_VAL == None:
            self.weights = [random() for _ in range(wn)]
            self.bias = random()

        self.INPTS_N = wn
        self.ACTIVATION = activation

        #
        self.latest_inputs = None

    def _get_output(self, vals):
        # VALS = vals.__len__()
        """
        VALS replaced with expected length self.INPTS_N
        to raise an error on incomplete input 
        and ignore the excess
        """

        # save the input
        self.latest_inputs = vals
        #

        rslt = 0

        for val_indx in range(self.INPTS_N):
            rslt += vals[val_indx]*self.weights[val_indx]

        rslt += self.bias

        return rslt

    # output

    def get_activated_output(self, vals):
        func = self.ACTIVATION
        if not func:
            func = Funcs.LINEAR

        return func(self._get_output(vals))

    # backprop

    def back_prop(self, expct):
        ERR = self.get_activated_output(self.latest_inputs) - expct

        # weight influences
        WEIGHTS_INF = self.latest_inputs
        for w_i in WEIGHTS_INF:
            w_i = ERR/w_i

            print(f"w_i {w_i}")

        # inputs influences
        INPTS_INF = self.weights
        for i_i in INPTS_INF:
            i_i = ERR/i_i

            print(f"i_i {i_i}")

        # bias influence
        b_i = ERR
        
