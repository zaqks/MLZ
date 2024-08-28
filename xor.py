from lib import *


ntwrk = Network(layers=[
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
], loss=Mse())


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


ntwrk.import_params()

ntwrk.train(X, Y, epochs=10000)
ntwrk.export_params()

ntwrk.run(X)
