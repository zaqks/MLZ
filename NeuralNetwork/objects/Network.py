from .Layer import Layer


class Network:
    # specify the inpt and out layer nodes num
    # for the hidden specify a list with their ndde counts

    def __init__(self, inpt, hidden, out):
        self.__inpt = inpt
        self.__hidden = hidden
        self.__out = out

        self.__neuronIndx = 0
        self.__layerIndx = 0
        self.__layers = []

        self.init_network()

    def add_layer(self, n):
        self.__layers.append(Layer(n, self.__layerIndx))
        self.__layerIndx += 1

    def init_network(self):
        # create the inpt layer
        self.add_layer(self.__inpt)
        # Create the hidden layers
        for i in self.__hidden:
            self.add_layer(i)
        # create the output layer
        self.add_layer(self.__out)

        # name the nodes
        beforeCnt = 0
        for i in self.__layers:
            for j in i.get_neurons():
                j.id = self.__neuronIndx

                self.__neuronIndx += 1

                # init the node
                if beforeCnt:
                    j.init_node(beforeCnt)

            beforeCnt = i.count_neurons()

    def print(self):
        for i in self.__layers:
            layer = ""
            for j in i.get_neurons():
                layer += str(f"{j.id} ")

            print(layer + "\n")

        print("\n")

        for i in self.__layers:
            layer = ""
            for j in i.get_neurons():
                layer += f"{j.weights} {j.bias}  "

            print(layer + "\n")
