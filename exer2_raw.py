import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import sys

# ---------- Modelo de Perceptrón ----------
class Perceptron:
    def __init__(self, input_dim, lr=0.01, epochs=100):
        self.w = np.random.randn(input_dim + 1)
        self.lr = lr
        self.epochs = epochs
        self.train_errors = []

    def predict(self, x):
        x = np.insert(x, 0, 1)
        return 1 if np.dot(self.w, x) >= 0 else -1

    def fit(self, X, y):
        for _ in range(self.epochs):
            errores = 0
            for xi, target in zip(X, y):
                xi_bias = np.insert(xi, 0, 1)
                pred = self.predict(xi)
                if pred != target:
                    self.w += self.lr * target * xi_bias
                    errores += 1
            self.train_errors.append(errores / len(y))

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

def evaluar(modelo, X, y):
    errores = 0
    for xi, target in zip(X, y):
        pred = modelo.predict(xi)
        if pred != target:
            errores += 1
    return errores / len(y)

#Entreno el 80% de mi dataset y el 20% restante lo uso para testear
def dividir_train_test(X, y, train_ratio=0.8):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    train_size = int(len(X) * train_ratio)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]

def main(save_file = False):
    data = np.loadtxt('TP3-ej2-conjunto.csv', delimiter=',', skiprows=1)

    X = data[:, :-1]
    y = data[:, -1]

    umbral = 30
    y = np.where(y > umbral, 1, -1)
    X_train, y_train, X_test, y_test = dividir_train_test(X, y)

    model_lineal = Perceptron(input_dim=X_train.shape[1], lr=0.01, epochs=100)
    model_lineal.fit(X_train, y_train)

    # -- Perceptrón No Lineal --
    X_train_nl = transformacion_no_lineal(X_train)
    X_test_nl = transformacion_no_lineal(X_test)

    model_nolineal = Perceptron(input_dim=X_train_nl.shape[1], lr=0.01, epochs=100)
    model_nolineal.fit(X_train_nl, y_train)

    train_error_lineal = evaluar(model_lineal, X_train, y_train)
    test_error_lineal = evaluar(model_lineal, X_test, y_test)
    train_error_nolineal = evaluar(model_nolineal, X_train_nl, y_train)
    test_error_nolineal = evaluar(model_nolineal, X_test_nl, y_test)

    output_data = {
        "train_error_lineal": round(train_error_lineal, 6),
        "test_error_lineal": round(test_error_lineal, 6),
        "train_error_no_lineal": round(train_error_nolineal, 6),
        "test_error_no_lineal": round(test_error_nolineal, 6)
    }

    # ---------- Graficar curvas de aprendizaje ----------

    if save_file:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs('results', exist_ok=True)
        root ='results/results_ex2_'+timestamp
        os.makedirs(root)
        filename = f'{root}/learning_curves.png'



        # Nombre del archivo
        output_filename = f'{root}/output_ex2.json'

        # Guardar el JSON
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)


        plt.figure(figsize=(10, 6))
        plt.plot(model_lineal.train_errors, label='Perceptrón Lineal')
        plt.plot(model_nolineal.train_errors, label='Perceptrón No Lineal')
        plt.xlabel('Época')
        plt.ylabel('Error de entrenamiento')
        plt.title('Curva de aprendizaje')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        # plt.show()

    return output_data


if __name__ == "__main__":
    # Leer argumento por terminal
    save_file = False
    if len(sys.argv) >= 2:
        save_file_arg = sys.argv[1].lower()
        if save_file_arg not in ['true', 'false']:
            print("El argumento para ver si se guarda la imagen debe ser 'true' o 'false'.")
            sys.exit(1)
        save_file = save_file_arg == 'true'

    main(save_file)
