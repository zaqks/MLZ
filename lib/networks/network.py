from ..layers.dense_layer import Dense
from ..layers.convolutional_layer import Convolutional
from ..layers.activations.activation_base import ActivationLayer

from ..optimizers.optimizers import GradientDescent, MomentumOptimizer

from orjson import loads, dumps
import matplotlib.pyplot as plt
import numpy as np

from os import environ

class Network:
    def __init__(self, layers, loss, optimizer=GradientDescent):
        """
        if dense_inpts_szs.__len__() != activations.__len__():
           raise Exception("Network not properly defined")

        dense = [Dense(dense_inpts_szs[i], dense_inpts_szs[i + 1])
                 for i in range(len(dense_inpts_szs) - 1)]
        """
        #
        self.layers = layers
        self.loss = loss()

        # opt
        for _ in self.layers:
            if not isinstance(_, ActivationLayer):
                # opt =  optimizer
                # print(opt)
                _.optimizer = optimizer()
                # init                                
                _.optimizer.init_vs(_.weights, _.biases, _.kernels)                

    def train(self, X, Y, a=0.1, epochs=10000, plot=False):
        # for plot
        if plot:            
            environ['QT_QPA_PLATFORM'] = 'xcb'
            fig, ax = plt.subplots()

            ax.set_xlim(0, epochs)
            ax.set_ylim(0, 1)

            line, = ax.plot([], [], lw=2)

            plt.xlabel("epch")
            plt.ylabel("err")
            plt.title(f"training a={a}")

            
            X_axis = []
            Y_axis = []

        #
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

            if plot:
                X_axis.append(e)
                Y_axis.append(error)

                line.set_data(X_axis, Y_axis)
                plt.draw()
                plt.pause(1e-3)

        if plot:
            plt.show()

    def run(self, x):
        # for x in X:
        # forward
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        # print(output)
        return output

    def export_params(self, path="export.json"):
        out = []
        for lyr in self.layers:

            if isinstance(lyr, Dense):
                _ = lyr.weights.tolist()
                out.append((_, lyr.biases.tolist()))
            elif isinstance(lyr, Convolutional):
                _ = lyr.kernels.tolist()
                out.append((_, lyr.biases.tolist()))

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
                    _, biases = data[indx]
                    _ = np.array(_)
                    biases = np.array(biases)

                    lyr.weights = _
                    lyr.biases = biases
                    indx += 1

                elif isinstance(lyr, Convolutional):
                    _, biases = data[indx]
                    _ = np.array(_)
                    biases = np.array(biases)

                    lyr.kernels = _
                    lyr.biases = biases
                    indx += 1

            print("params import success")

        except:
            print("params import error")
