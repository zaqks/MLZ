<img src="docs/logo.svg" width="100%">


# Machine Learning library
MLZ is a lightweight machine learning library built from scratch using NumPy. It includes implementations of sequential and convolutional neural networks, along with common loss functions and optimizers like Adam. Key features include being NumPy-only, containing layers like Dense, Conv2D, and ReLU, and offering optimizers such as SGD and Adam, all with a modular design to facilitate experimentation and readability.

<h2>Example 1 (XOR)</h2>

```py
from lib.networks.network import *

from lib.layers.dense_layer import Dense
from lib.layers.convolutional_layer import Convolutional
from lib.layers.reshape_layer import Reshape

from lib.layers.activations.activation_layer import *
from lib.losses.loss import *

from lib.optimizers.optimizers import *

# the imports are relative to the lib dir for testing purposes ONLY, otherwise import the module since it's defined in __init__.py


ntwrk = Network(layers=[
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
], loss=Mse, optimizer=AdamOptimizer)


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


# import params
ntwrk.import_params()
# ntwrk.export_params()

# train
ntwrk.train(X, Y, epochs=100, a=0.055,  plot=True, plot_save="docs/figures/")
# ntwrk.export_params()

# run to test
ntwrk.run(np.reshape([[1, 0]], (2, 1)))

```



