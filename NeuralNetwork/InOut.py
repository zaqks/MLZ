from .Network import Network
from orjson import loads, dumps


"""
the way it works is by storing a list of sparate
weights and biases for each neuron

"""


class InOut:
    def __init__(self, network):
        self.network = network

    def import_data(self, path="import.json"):
        with open(path, "rb") as f:
            data = loads(f.read())
            f.close()

        #
        WEIGHTS = data["weights"]
        BIASES = data["biases"]

        indx_b = 0
        indx_w = 0

        for neuron in self.network.get_neurons():
            # set the bias
            neuron.bias = BIASES[indx_b]
            indx_b += 1

            # set the weights
            for _w in range(neuron.INPTS_N):
                neuron.weights[_w] = WEIGHTS[indx_w]
                indx_w += 1

    def export_data(self, path="export.json"):
        NEURONS = self.network.get_neurons()

        WEIGHTS = [val for nrn in NEURONS for val in nrn.weights]
        BIASES = [nrn.bias for nrn in NEURONS]

        OUT = {
            "weights": WEIGHTS,
            "biases": BIASES
        }

        with open(path, "wb") as f:
            f.write(dumps(OUT))
            f.close()
