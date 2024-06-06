from random import random
from ..funcs import ActvFuncs


RND = 1  # rounding
W, B = 100, 10  # w*=W, b*=B


class Neuron:
    def __init__(self, weights, bias, actv=ActvFuncs.RELU, id=-1):
        self.id = id

        self.weights = weights
        self.bias = bias

        self.__actv = actv

    def set_weights(self, w):
        self.weights = w

    def set_bias(self, b):
        self.bias = b

    def set_neuron(self, w, b):
        self.set_weights(w)
        self.set_bias(b)

    def init_node(self, wn):
        wghts = []

        for i in range(wn):
            wghts.append(random() * W)
            # wghts.append(round(random(), RND) * W)

        self.set_weights(wghts)
        self.set_bias(random() * B)
        # self.set_bias(round(random(), RND) * B)

    def activation_func(self, x):
        return ActvFuncs.get(self.__actv)(x)

    def output(self, inpt):
        rslt = 0
        for i in range(inpt.__len__()):
            rslt += inpt[i] * self.weights[i]

        return self.activation_func(rslt + self.bias)
