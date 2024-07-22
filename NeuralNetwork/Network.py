from .Layer import Layer, Funcs


class Network:
    def __init__(self, layers_ns, activation=None):
        self.layers = []

        previous_layer_ns = layers_ns[0]
        # self.layers.append(Layer(previous_layer_ns, previous_layer_ns))

        for cl, layer_ns in enumerate(layers_ns[1:]):
            self.layers.append(
                Layer(previous_layer_ns, layer_ns, activation, cl=cl))
            previous_layer_ns = layer_ns

    def forward_propg(self, inpt):
        previous = inpt
        for layer in self.layers:
            previous = layer.get_activated_outputs(previous)

        return previous

    def get_neurons(self):
        return [

            nrn for lyr in self.layers for nrn in lyr.get_neurons()

        ]

    def print_neurons(self):
        for neuron in self.get_neurons():
            print("\n")
            print(f"{neuron.weights}")
            print(f"{neuron.bias}")

    def backward_propg(self, expct  # , rslt=None
                       ):
        # if rslt:
        #    print(f"rslt: {rslt}")
        #    print(f"expct: {expct}")
        #    print("-----------------")

        for i, nrn in enumerate(self.layers[-1].get_neurons()):
            nrn.back_prop(expct[i], self.layers)
