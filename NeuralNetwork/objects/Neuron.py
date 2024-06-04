from random import random


RND = 1  # rounding


class Neuron:
    def __init__(self, weights, bias, id=-1):
        self.id = id

        self.weights = weights
        self.bias = bias

    def output(self, inpt):
        rslt = 0
        for i in range(inpt.__len__()):
            rslt += inpt[i] * self.weights[i]

        return rslt + self.bias

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
            wghts.append(round(random(), RND))

        self.set_weights(wghts)
        self.set_bias(round(random(), RND))


    