from .ActivationFunctions import RELU, LINEAR


class Neuron:
    def __init__(self, wn):
        self.weights = [0 for _ in range(wn)]
        self.bias = 0

    def _get_output(self, vals):
        VALS = vals.__len__()
        rslt = 0

        for val_indx in range(VALS):
            rslt += vals[val_indx]*self.weights[val_indx]

        rslt += self.bias

        return rslt

    def get_activated_output(self, vals, func=RELU):
        return func(self._get_output(vals))
