from lib.dense_layer import Dense
from lib.activation_layer import TanH
from lib.loss import mse, mse_prime
import numpy as np


# dataset
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


# network
network = [
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
    TanH()
]


epochs = 10000
learning_rate = 0.1


for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # err
        error += mse(y, output)

        # back
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print(f"{e}/{epochs} error={error}")
