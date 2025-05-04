import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys
from matplotlib import cm
from perceptron import Perceptron
from utils import dividir_train_test, evaluar, transformacion_no_lineal


def entrenar_no_lineal(X, y, lr=0.005, epochs=100):
    X_nl = transformacion_no_lineal(X)
    X_train, y_train, X_test, y_test = dividir_train_test(X_nl, y)
    model = Perceptron(input_dim=X_nl.shape[1], lr=lr, epochs=epochs)
    model.fit(X_train, y_train)
    train_error = evaluar(model, X_train, y_train)
    test_error = evaluar(model, X_test, y_test)
    return model, train_error, test_error


# def evaluar_generalizacion_no_lineal(X, y, repeticiones=10, lr=0.005, epochs=100):
#     test_errores = []
#     for _ in range(repeticiones):
#         _, _, test_error = entrenar_no_lineal(X, y, lr, epochs)
#         test_errores.append(test_error)
#     promedio = np.mean(test_errores)
#     desvio = np.std(test_errores)
#     return promedio, desvio


def cross_validation_no_lineal(X, y, k=5, lr=0.005, epochs=100):
    X_nl = transformacion_no_lineal(X)
    n = len(X_nl)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    test_errors = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X_nl[train_idx], X_nl[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Perceptron(input_dim=X_nl.shape[1], lr=lr, epochs=epochs)
        model.fit(X_train, y_train)
        error = evaluar(model, X_test, y_test)
        test_errors.append(error)

    promedio = np.mean(test_errors)
    desvio = np.std(test_errors)
    return promedio, desvio

def evaluar_generalizacion_cross_val(X, y, repeticiones=50, lr=0.005, epochs=500):
    errores = []
    for _ in range(repeticiones):
        avg, _ = cross_validation_no_lineal(X, y, k=5, lr=lr, epochs=epochs)
        errores.append(avg)
    return errores


def main(save_file=False):
    data = np.loadtxt('TP3-ej2-escalado.csv', delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    lr_no_lineal = 0.005
    epochs = [10,20,50,100,200,500]
    # epochs = [200,500,1000,2000,3000,5000,10000]
    resultados = []
    # test_avg, test_std = evaluar_generalizacion_no_lineal(X, y, repeticiones=100, lr=lr_no_lineal, epochs=100)
    # for epoch in epochs:
    #     print(f"Evaluando epochs: {epoch}")
    #     cv_avg, cv_std = cross_validation_no_lineal(X, y, k=5, lr=lr_no_lineal, epochs=epoch)
    #     resultados.append((epoch, cv_avg, cv_std))
    
    cv_avg, cv_std = cross_validation_no_lineal(X, y, k=5, lr=lr_no_lineal, epochs=500)
    output_data = {
        # "no_lineal_generalization_avg": round(test_avg, 6),
        # "no_lineal_generalization_std": round(test_std, 6),
        "no_lineal_cv_avg": round(cv_avg, 6),
        "no_lineal_cv_std": round(cv_std, 6)
    }

    # epocas = [r[0] for r in resultados]
    # promedios = [r[1] for r in resultados]
    # desvios = [r[2] for r in resultados]




    errores_generales = evaluar_generalizacion_cross_val(X, y, repeticiones=100, lr=lr_no_lineal, epochs=500)
        
    plt.figure(figsize=(12,5))

        # Boxplot
    plt.subplot(1, 2, 1)
    plt.boxplot(errores_generales, vert=True, patch_artist=True)
    plt.title('Distribución del Error de Validación Cruzada (500 épocas)')
    plt.ylabel('Error')
    plt.grid(True)

        # Histograma
    plt.subplot(1, 2, 2)
    counts, bins, patches = plt.hist(errores_generales, bins=15, edgecolor='black')

    norm = plt.Normalize(bins.min(), bins.max())
    cmap = cm.plasma

    for count, bin_left, patch in zip(counts, bins, patches):
        bin_center = bin_left + (bins[1] - bins[0]) / 2
        color = cmap(norm(bin_center))
        patch.set_facecolor(color)
    plt.title('Histograma de Errores (500 épocas)')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


    if save_file:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        root = f'results/results_ex2_no_lineal_{timestamp}'
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
