# Machine Learning library


# Example 1

```
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



