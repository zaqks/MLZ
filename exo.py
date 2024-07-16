from NeuralNetwork import Network, InOut, NetworkFuncs


ntwrk = Network([2, 2,  1], NetworkFuncs.LINEAR)

io = InOut(ntwrk)
io.import_data("exo.json")


for i in range(4):

    rslt = ntwrk.forward_propg([0.35, 0.9])
    print(rslt)

    ERR = rslt[0]-0.5
    print(f"error {ERR}\n")

    ntwrk.backward_propg([0.5])

    print("------------------")
    

io.export_data("exo_export.json")
