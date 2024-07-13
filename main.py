from NeuralNetwork import *


ntwrk = Network([2,  1], NetworkFuncs.RELU)
inout = InOut(ntwrk)
inout.import_data()

rslt = ntwrk.forward_probg([2, 3])
print(rslt)


inout.export_data()



