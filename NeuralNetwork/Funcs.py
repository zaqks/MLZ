from math import exp, log


class Funcs:

    def SOFTMAX(inpt):
        exps = [exp(i) for i in inpt]
        exps_sum = sum(exps)

        rslt = [i/exps_sum for i in exps]

        return rslt

        # Activation Functions

    def LINEAR(val):
        return val

    def RELU(val):
        return max(0, val)

    def SIGMOID(x):
        return 1 / (1 + exp(-x))
