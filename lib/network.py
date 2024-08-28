from .dense_layer import Dense
from orjson import loads, dumps

import numpy as np


class Network:
    def __init__(self, layers, loss):
        """
        if dense_inpts_szs.__len__() != activations.__len__():
           raise Exception("Network not properly defined")

        dense = [Dense(dense_inpts_szs[i], dense_inpts_szs[i + 1])
                 for i in range(len(dense_inpts_szs) - 1)]
        """
        #
        self.layers = layers
        #for layer, activation in zip(dense, activations):
        #    self.layers.extend([layer, activation])

        #
        self.loss = loss

    def train(self, X, Y, a=0.1, epochs=10000):
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                # forward
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                # back
                grad = self.loss.prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, a)

                # err
                error += self.loss.func(y, output)

            error /= len(X)
            print(f"{e+1}/{epochs} error={error}")

    def run(self, X):
        for x in X:
            # forward
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            print(output)

    def export_params(self, path="export.json"):
        out = []
        for lyr in self.layers:
            if isinstance(lyr, Dense):
                out.append((lyr.weights.tolist(), lyr.bias.tolist()))

        with open(path, "wb") as f:
            f.write(dumps(out))
            f.close()

    def import_params(self, path="export.json"):
        try:
            with open(path, "rb") as f:
                data = loads(f.read())
                f.close()

            indx = 0
            for lyr in self.layers:
                if isinstance(lyr, Dense):
                    weights, bias = data[indx]
                    indx += 1
                    #
                    lyr.weights = np.array(weights)
                    lyr.bias = np.array(bias)

        except:
            print("import error")
