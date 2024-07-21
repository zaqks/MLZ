from .Funcs import Funcs
from random import random

INIT_VAL = 1


class Correction:
    def __init__(self, val):
        self.abs_val = abs(val)
        self.val = val


class Corrections:
    def __init__(self, lst):
        self.lst = lst

    def get_max_correction(self):
        abs_vals = [crct.abs_val for crct in self.lst]
        indx = self.abs_vals.index(max(abs_vals))

        return lst[indx]


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
        # ERR = self.get_activated_output(self.latest_inputs) - expct
        ERR = self._get_output(self.latest_inputs) - expct

        # bias correction
        BIAS = Correction(ERR)

        # weight corrections
        WEIGHTS = [Correction(ERR/i if i else 0) for i in self.latest_inputs]

        # inputs corrections
        INPTS = [Correction(ERR/i if i else 0)
                 for i in self.weights]

        # print(f"BIAS CORRECTION {BIAS}")
        # print(f"WEIGHTS CORRECTION {WEIGHTS}")
        # print(f"INPTS CORRECTION {INPTS}")

        # get the max correction to see what to do
        max_inpt = Corrections(INPTS).get_max_correction()
        max_weight = Corrections(WEIGHTS).get_max_correction()

        # go back to the previous layer
        # or no previous layer
        BACK = (False not in [max_abs_inpt > _ for _ in [
                max_abs_weight, BIAS.abs_val]]) and current_layer_indx > 0

        if not BACK:
            if BIAS.abs_val > max_abs_weight:
                # max_weight = max_weight if max_weight == max_abs_weight else -max_abs_weight
                # update the bias
                self.bias -= BIAS
            else:
                # update the weight
                max_weight = max_weight if max_weight == max_abs_weight else -max_abs_weight

                self.weights[WEIGHTS.index(
                    max_weight)] -= max_weight

        else:
            max_inpt = max_inpt if max_inpt == max_abs_inpt else -max_abs_inpt

            inpt_indx = INPTS.index(max_inpt)
            target_nrn = layers[current_layer_indx -
                                1].get_neurons()[inpt_indx]
            target_nrn .back_prop(
                self.latest_inputs[inpt_indx] - max_inpt, layers, current_layer_indx-1)
