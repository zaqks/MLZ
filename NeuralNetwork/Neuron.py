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

    def back_prop(self, expct, layers, current_layer_indx):
        if current_layer_indx < 0:
            return

        # ERR = self.get_activated_output(self.latest_inputs) - expct
        ERR = self._get_output(self.latest_inputs) - expct

        BIAS = ERR  # bias correction
        WEIGHTS = [ERR/i for i in self.latest_inputs]  # weight corrections
        INPTS = [ERR/i for i in self.weights]  # inputs corrections

        print(f"BIAS CORRECTION {BIAS}")
        print(f"WEIGHTS CORRECTION {WEIGHTS}")
        print(f"INPTS CORRECTION {INPTS}")

        # get the max to see what to do

        max_inpt = max(INPTS)
        max_weight = max(WEIGHTS)

        # go back to the previous layer
        # or no previous layer
        BACK = (False not in [max_inpt > _ for _ in [
                max_weight, BIAS]]) and current_layer_indx > 0

        if not BACK:
            if BIAS > max_weight:
                # update the bias
                self.bias -= BIAS
            else:
                # update the weight
                self.weights[WEIGHTS.index(max_weight)] -= max_weight

        else:
            inpt_indx = INPTS.index(max_inpt)
            target_nrn = layers[current_layer_indx -
                                1].get_neurons()[inpt_indx]
            target_nrn .back_prop(
                self.latest_inputs[inpt_indx] - max_inpt, layers, current_layer_indx-1)
