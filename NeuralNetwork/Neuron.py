from .Funcs import Funcs
from random import random
import math

INIT_VAL = None


class Correction:
    def __init__(self, val):
        self.abs_val = abs(val)
        self.val = val


class Corrections:
    def __init__(self, lst):
        self.lst = lst

    def get_max_correction(self):
        abs_vals = [crct.abs_val for crct in self.lst]
        indx = abs_vals.index(max(abs_vals))

        return self.lst[indx]


class Neuron:
    def __init__(self, wn,   activation=None, rw=None, cl=None,):
        self.weights = [INIT_VAL for _ in range(wn)]
        self.bias = INIT_VAL

        if INIT_VAL == None:
            self.weights = [random() for _ in range(wn)]
            self.bias = random()

        self.INPTS_N = wn
        self.ACTIVATION = activation

        #
        self.latest_inputs = None

        # Row Column
        self.rw = rw
        self.cl = cl

        #
        self.weight_changes = []
        self.bias_changes = []

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

    def change_collect(self, expct, layers):
        # Calibrate the previous layer to keep constant inpts
        # get the expected value which is our new neuron output
        # calibrate the next layer's weights comming from this neuron
        # expect the neuron we came from

        # chouf the neuron u came from first X
        CURRENT_OUTPUT = self._get_output(self.latest_inputs)

        # ERR = math.pow(CURRENT_OUTPUT - expct, 2)
        # ERR = math.exp(2 * math.log(CURRENT_OUTPUT - expct))

        ERR = CURRENT_OUTPUT - expct

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
        max_bias = Correction(ERR)  # the max is just a name here

        # go back to the previous layer
        # or no previous layer
        BACK = (False not in [max_inpt.abs_val > _ for _ in [
                max_weight.abs_val, max_bias.abs_val]]) and self.cl > 0

        if not BACK:
            if max_bias.abs_val > max_weight.abs_val:
                # max_weight = max_weight if max_weight == max_abs_weight else -max_abs_weight
                # update the bias
                # self.bias -= max_bias.val
                self.push_bias_change(max_bias.val)

            else:
                # update the weight
                change = [0 for _ in range(self.weights.__len__())]

                change[WEIGHTS.index(
                    max_weight)] = max_weight.val

                self.push_weight_change(change)

        else:
            inpt_indx = INPTS.index(max_inpt)
            target_nrn = layers[self.cl -
                                1].get_neurons()[inpt_indx]
            target_nrn .change_collect(
                self.latest_inputs[inpt_indx] - max_inpt.val, layers)

    def push_weight_change(self, val):
        self.weight_changes.append(val)

    def push_bias_change(self, val=0):
        self.bias_changes.append(val)

    def clear_changes(self):
        self.weight_changes.clear()
        self.bias_changes.clear()

    def apply_changes(self):
        # get the averages
        bias_change = sum(
            self.bias_changes)/self.bias_changes.__len__() if self.bias_changes.__len__() else 0

        weights_change = [0 for _ in range(self.weights.__len__())]

        for vect in self.weight_changes:
            for indx, val in enumerate(vect):
                weights_change[indx] += val

        weights_change = [_/self.weight_changes.__len__() if self.weight_changes.__len__() else 0
                          for _ in weights_change]

        # apply the changes
        self.bias -= bias_change

        for indx, val in enumerate(weights_change):
            self.weights[indx] -= val

        # clear everything

        self.clear_changes()
