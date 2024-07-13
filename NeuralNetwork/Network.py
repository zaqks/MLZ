from .Layer import Layer, Funcs


class Network:
    def __init__(self, layers_ns, activation=None):
        self.layers = []

        previous_layer_ns = layers_ns[0]
        #self.layers.append(Layer(previous_layer_ns, previous_layer_ns))

        for layer_ns in layers_ns[1:]:
            self.layers.append(Layer(previous_layer_ns, layer_ns, activation))
            previous_layer_ns = layer_ns

    def forward_probg(self, inpt):
        previous = inpt
        for layer in self.layers:
            previous = layer.get_activated_outputs(previous)                        

        return previous
