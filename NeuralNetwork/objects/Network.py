from .Layer import Layer


class Network:
    # specify the inpt and out layer nodes num
    # for the hidden specify a list with their ndde counts

    def __init__(self, inpt, hidden, out):
        self.__layerIndx = 0
        self.__layers = []

        # create the inpt layer
        self.add_layer(inpt)
        # Create the hidden layers
        for i in hidden:
            self.add_layer(i)
        # create the output layer
        self.add_layer(out)

    def add_layer(self, n):
        self.__layers.append(Layer(n, self.__layerIndx))
        self.__layerIndx += 1

    def print(self):
        for i in self.__layers:
            layer = ""
            for j in i.get_neurons():
                layer += str(f"{j.id} ")

            print(layer + "\n")