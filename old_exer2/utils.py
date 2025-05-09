import numpy as np

def dividir_train_test(X, y, train_ratio=0.8):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    train_size = int(len(X) * train_ratio)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]


def transformacion_no_lineal(X):
    return np.sinh(X)

def evaluar(modelo, X, y):
    max_valor = max(np.abs(X).max(), np.abs(y).max())
    errores = 0
    for xi, target in zip(X, y):
        pred = modelo.predict(xi)
        if pred != target:
            errores += abs(target - pred)/max_valor
    return errores / len(y)
