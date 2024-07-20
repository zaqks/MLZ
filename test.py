from NeuralNetwork import Network,  NetworkFuncs, InOut


ntwr = Network([3, 1], activation=NetworkFuncs.LINEAR)

io = InOut(ntwr)
io.import_data("data/import.json")


rslt = ntwr.forward_propg([0.1, 0.2, 0.3])
print(rslt)


ntwr.backward_propg([1])
io.export_data("data/import.json")
