from math import exp

NRM = True  # normalize


def normalize(x):
    while x > 1:
        x /= 10
    return x


def sigmoid(x):
    return 1 / (1 + exp(-x))


def relu(x):
    x = max(0, x)
    if NRM:
        x = normalize(x)

    return x


def linear(x):
    if NRM:
        x = normalize(x)

    return x
