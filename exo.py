from NeuralNetwork import Network, InOut, NetworkFuncs


ntwrk = Network([2, 2, 1], NetworkFuncs.SIGMOID)

io = InOut(ntwrk)
io.import_data("exo.json")

rslt = ntwrk.forward_probg([0.35, 0.9])
print(rslt)

#io.export_data("exo.json")

