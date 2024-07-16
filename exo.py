from NeuralNetwork import Network, InOut, NetworkFuncs


ntwrk = Network([2, 2,  1], NetworkFuncs.SIGMOID)

io = InOut(ntwrk)
io.import_data("exo.json")


for i in range(20):

    rslt = ntwrk.forward_propg([0.35, 0.9])
    ntwrk.backward_propg([0.5], rslt)


io.export_data("exo_export.json")
