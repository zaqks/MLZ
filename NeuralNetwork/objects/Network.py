from .Layer import Layer, ActvFuncs
from orjson import dumps, loads


class Network:
    # specify the inpt and out layer nodes num
    # for the hidden specify a list with their ndde counts

    def __init__(self, inpt, hidden, out,actv=ActvFuncs.RELU,  src=None):
        self.__inpt = inpt
        self.__hidden = hidden
        self.__out = out

        self.__neuronIndx = 0
        self.__layerIndx = 0
        self.__layers = []

        self.__actv = actv

        self.init_network(src)

    def add_layer(self, n):
        self.__layers.append(Layer(n, self.__layerIndx, self.__actv))
        self.__layerIndx += 1

    def init_network(self, src=None):
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
                if beforeCnt and not src:
                    j.init_node(beforeCnt)

            beforeCnt = i.count_neurons()

        # import the params
        if src:
            with open(src, "rb") as f:
                src = loads(f.read())
                f.close()

            for i in self.__layers:
                layer_id = f"layer_{i.id}"

                indx = 0
                nrns = i.get_neurons()
                for j in range(nrns.__len__()):
                    nrn = nrns[j]

                    neuron_id = f"neuron_{nrn.id}"

                    srcNrn = src[layer_id][neuron_id]
                    nrn.weights = srcNrn[0]
                    nrn.bias = srcNrn[1]

    def show_ids(self):
        for i in self.__layers:
            layer = ""
            for j in i.get_neurons():
                layer += str(f"{j.id} ")

            print(layer)
        print("\n")

    def show_params(self):
        for i in self.__layers:
            layer = ""
            for j in i.get_neurons():
                layer += f"{j.weights} {j.bias}  "

            print(layer)

        print("\n")

    def frwrd_prbg(self, inpt):
        # feed each output to the input

        prev = inpt
        for indx in range(1, self.__layers.__len__()):
            layer = self.__layers[indx]
            prev = layer.output(prev)

        return prev

    def export(self, filename="json/out.json"):
        rslt = {}

        for i in self.__layers:
            layer_id = f"layer_{i.id}"
            rslt[layer_id] = {}
            for j in i.get_neurons():
                neuron_id = f"neuron_{j.id}"
                rslt[layer_id][neuron_id] = [j.weights, j.bias]

        with open(filename, "wb") as f:
            f.write(dumps(rslt))
            f.close()
        pass
