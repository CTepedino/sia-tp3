import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys

from exer2_perceptron import Perceptron
from exer2_utils import dividir_train_test, transformacion_no_lineal, evaluar

def entrenar_lineal(X, y, lr=0.01, epochs=100):
    X_train, y_train, X_test, y_test = dividir_train_test(X, y)
    model = Perceptron(input_dim=X.shape[1], lr=lr, epochs=epochs)
    model.fit(X_train, y_train)
    train_error = evaluar(model, X_train, y_train)
    test_error = evaluar(model, X_test, y_test)
    return model, train_error, test_error

def entrenar_no_lineal(X, y, lr=0.01, epochs=100):
    X_nl = transformacion_no_lineal(X)
    X_train, y_train, X_test, y_test = dividir_train_test(X_nl, y)
    model = Perceptron(input_dim=X_nl.shape[1], lr=lr, epochs=epochs)
    model.fit(X_train, y_train)
    train_error = evaluar(model, X_train, y_train)
    test_error = evaluar(model, X_test, y_test)
    return model, train_error, test_error

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

    model_lineal, train_error_l, test_error_l = entrenar_lineal(X, y)
    model_nolineal, train_error_nl, test_error_nl = entrenar_no_lineal(X, y)

    test_avg, test_std = evaluar_generalizacion_lineal(X, y)

    output_data = {
        "train_error_lineal": round(train_error_l, 6),
        "test_error_lineal": round(test_error_l, 6),
        "train_error_no_lineal": round(train_error_nl, 6),
        "test_error_no_lineal": round(test_error_nl, 6),
        "lineal_generalization_avg": round(test_avg, 6),
        "lineal_generalization_std": round(test_std, 6)
    }

    if save_file:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        root = f'results/results_ex2_{timestamp}'
        os.makedirs(root)

        with open(f'{root}/output_ex2.json', 'w') as f:
            json.dump(output_data, f, indent=4)

        plt.figure(figsize=(10, 6))
        plt.plot(model_lineal.train_errors, label='Lineal')
        plt.plot(model_nolineal.train_errors, label='No Lineal')
        plt.xlabel('Época')
        plt.ylabel('Error de entrenamiento')
        plt.title('Curvas de aprendizaje')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{root}/learning_curves.png')

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
