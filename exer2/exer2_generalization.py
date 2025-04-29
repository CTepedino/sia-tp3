import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys

from exer2_perceptron import Perceptron
from exer2_utils import dividir_train_test, evaluar


# se divide en k bloques
# lr es el learning rate
# epochs es el número de épocas para el entrenamiento
def cross_validation_lineal(X, y, k=5, lr=0.01, epochs=100):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    test_errors = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Perceptron(input_dim=X.shape[1], lr=lr, epochs=epochs)
        model.fit(X_train, y_train)
        # revisar si evaluar devuelve el error o q devuelve
        error = evaluar(model, X_test, y_test)
        test_errors.append(error)

    promedio = np.mean(test_errors)
    desvio = np.std(test_errors)
    return promedio, desvio


def entrenar_lineal(X, y, lr=0.01, epochs=100):
    X_train, y_train, X_test, y_test = dividir_train_test(X, y)
    model = Perceptron(input_dim=X.shape[1], lr=lr, epochs=epochs)
    model.fit(X_train, y_train)
    train_error = evaluar(model, X_train, y_train)
    test_error = evaluar(model, X_test, y_test)
    return model, train_error, test_error

# - No se generan datos nuevos.
# - Se divide el dataset original aleatoriamente en 80% entrenamiento y 20% testeo.
# - El modelo se entrena solo con el 80% → y se evalúa con el 20% restante, que nunca vio antes.
# - Se repite varias veces con diferentes divisiones (shuffles) para tener una idea estable del error.
# - Esto permite estimar cómo se comportaría el modelo con datos nuevos reales.


def evaluar_generalizacion_lineal(X, y, repeticiones=10, lr=0.01, epochs=100):
    test_errores = []
    for _ in range(repeticiones):
        _, _, test_error = entrenar_lineal(X, y, lr, epochs)
        test_errores.append(test_error)
    promedio = np.mean(test_errores)
    desvio = np.std(test_errores)
    return promedio, desvio

def main(save_file=False):
    data = np.loadtxt('TP3-ej2-conjunto.csv', delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = np.where(data[:, -1] > 30, 1, -1)

    test_avg, test_std = evaluar_generalizacion_lineal(X, y)
    output_data = {
        "lineal_generalization_avg": round(test_avg, 6),
        "lineal_generalization_std": round(test_std, 6)
    }

    cv_avg, cv_std = cross_validation_lineal(X, y)
    output_data.update({
        "lineal_cv_avg": round(cv_avg, 6),
        "lineal_cv_std": round(cv_std, 6)
    })

    if save_file:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        root = f'results/results_ex2_generalization_{timestamp}'
        os.makedirs(root)

        with open(f'{root}/output_ex2.json', 'w') as f:
            json.dump(output_data, f, indent=4)


    return output_data

if __name__ == "__main__":
    save_file = False
    if len(sys.argv) >= 2:
        arg = sys.argv[1].lower()
        if arg not in ['true', 'false']:
            print("El argumento debe ser 'true' o 'false'")
            sys.exit(1)
        save_file = arg == 'true'

    resultados = main(save_file)
    print(json.dumps(resultados, indent=4))
