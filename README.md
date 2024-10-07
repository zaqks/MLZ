<img src="https://github.com/zaqks/NeuralNetwork/blob/b11295b03b2e7b81d5386c4bbac8f9ce7b8e29cd/docs/logo.svg" width="100%">


# Machine Learning library (ML stands for MeLo)
# Training Plotting    

<h2>Example 1 (XOR)</h2>

```py
from lib.networks.network import *

from lib.layers.dense_layer import Dense
from lib.layers.convolutional_layer import Convolutional
from lib.layers.reshape_layer import Reshape

from lib.layers.activations.activation_layer import *
from lib.losses.loss import *

# the imports are relative to the lib dir for testing purposes ONLY, otherwise import the module since it's defined in __init__.py



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
```



