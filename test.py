from lib.networks.network import *

from lib.layers.dense_layer import Dense
from lib.layers.convolutional_layer import Convolutional
from lib.layers.reshape_layer import Reshape

from lib.layers.activations.activation_layer import *
from lib.losses.loss import *



ntwrk = Network(layers=[
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
], loss=Mse())


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


# import params
ntwrk.import_params()

# train
ntwrk.train(X, Y, epochs=100, plot=True)
ntwrk.export_params()

# run to test
ntwrk.run(np.reshape([[1, 0]], (2, 1)))