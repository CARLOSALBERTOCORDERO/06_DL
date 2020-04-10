import numpy as np

def layer_sizes_test():
    np.random.seed(1)
    X_test = np.random.randn(2, 3)
    Y_test = np.random.randn(1, 3)
    return X_test, Y_test