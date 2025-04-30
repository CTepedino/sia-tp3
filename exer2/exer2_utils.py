import numpy as np

def dividir_train_test(X, y, train_ratio=0.8):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    train_size = int(len(X) * train_ratio)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]

def transformacion_no_lineal(X):
    X_nl = []
    for x in X:
        x1, x2, x3 = x
        X_nl.append([
            x1, x2, x3,
            x1**2, x2**2, x3**2,
            x1*x2, x1*x3, x2*x3
        ])
    return np.array(X_nl)

#devuelve el modulo entre el objetivo y la prediccion dividido por el tamaño del conjunto de test
#para que de entre 0 y 1
def evaluar(modelo, X, y):
    max_valor = max(np.abs(X).max(), np.abs(y).max())
    errores = 0
    for xi, target in zip(X, y):
        pred = modelo.predict(xi)
        if pred != target:
            errores += abs(target - pred)/max_valor
    return errores / len(y)
