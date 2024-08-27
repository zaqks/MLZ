from lib import *


ntwrk = Network([2, 3,   1], [TanH(), TanH(), TanH()], Mse())


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


ntwrk.import_params()

#ntwrk.train(X, Y, epochs=20000)
#ntwrk.export_params()

ntwrk.run(X)
