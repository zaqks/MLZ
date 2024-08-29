<img src="https://github.com/zaqks/NeuralNetwork/blob/b11295b03b2e7b81d5386c4bbac8f9ce7b8e29cd/docs/logo.svg" width="100%">


# Machine Learning library (ML stands for MeLo)
    

<h2>Example 1 (XOR)</h2>

```py
from MLib import *

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
ntwrk.train(X, Y, epochs=10000)
ntwrk.export_params()

# run to test
ntwrk.run(np.reshape([[1, 0]], (2, 1)))
```



