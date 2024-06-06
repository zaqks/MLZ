from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def relu( x):
        return max(0, x)

def linear(x):
    return x