from .Funcs import Funcs

INIT_VAL = 1


class Neuron:
    def __init__(self, wn):
        self.weights = [INIT_VAL for _ in range(wn)]
        self.bias = INIT_VAL

        self.INPTS_N = wn

    def _get_output(self, vals):
        # VALS = vals.__len__()
        """
        VALS replaced with expected length self.INPTS_N
        to raise an error on incomplete input 
        and ignore the excess
        """

        rslt = 0

        for val_indx in range(self.INPTS_N):
            rslt += vals[val_indx]*self.weights[val_indx]

        rslt += self.bias

        return rslt

    # output

    def get_activated_output(self, vals, func=Funcs.LINEAR):

        return func(self._get_output(vals))

