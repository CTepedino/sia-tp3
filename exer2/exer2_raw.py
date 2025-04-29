# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# import json
# import sys

# # ---------- Modelo de Perceptrón ----------
# class Perceptron:
#     def __init__(self, input_dim, lr=0.01, epochs=100):
#         self.w = np.random.randn(input_dim + 1)
#         self.lr = lr
#         self.epochs = epochs
#         self.train_errors = []

#     def predict(self, x):
#         x = np.insert(x, 0, 1)
#         return 1 if np.dot(self.w, x) >= 0 else -1

#     def fit(self, X, y):
#         for _ in range(self.epochs):
#             errores = 0
#             for xi, target in zip(X, y):
#                 xi_bias = np.insert(xi, 0, 1)
#                 pred = self.predict(xi)
#                 if pred != target:
#                     self.w += self.lr * target * xi_bias
#                     errores += 1
#             self.train_errors.append(errores / len(y))

# def transformacion_no_lineal(X):
#     X_nl = []
#     for x in X:
#         x1, x2, x3 = x
#         X_nl.append([
#             x1, x2, x3,
#             x1**2, x2**2, x3**2,
#             x1*x2, x1*x3, x2*x3
#         ])
#     return np.array(X_nl)

# def evaluar(modelo, X, y):
#     errores = 0
#     for xi, target in zip(X, y):
#         pred = modelo.predict(xi)
#         if pred != target:
#             errores += 1
#     return errores / len(y)

# #Entreno el 80% de mi dataset y el 20% restante lo uso para testear
# def dividir_train_test(X, y, train_ratio=0.8):
#     idxs = np.arange(len(X))
#     np.random.shuffle(idxs)
#     train_size = int(len(X) * train_ratio)
#     train_idxs = idxs[:train_size]
#     test_idxs = idxs[train_size:]
#     return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]


# def entrenar_perceptron_lineal(X, y, lr=0.01, epochs=100):
#     X_train, y_train, X_test, y_test = dividir_train_test(X, y)
#     modelo = Perceptron(input_dim=X_train.shape[1], lr=lr, epochs=epochs)
#     modelo.fit(X_train, y_train)

#     train_error = evaluar(modelo, X_train, y_train)
#     test_error = evaluar(modelo, X_test, y_test)

#     return modelo, train_error, test_error

# def entrenar_perceptron_no_lineal(X, y, lr=0.01, epochs=100):
#     X_nl = transformacion_no_lineal(X)
#     X_train, y_train, X_test, y_test = dividir_train_test(X_nl, y)
#     modelo = Perceptron(input_dim=X_train.shape[1], lr=lr, epochs=epochs)
#     modelo.fit(X_train, y_train)

#     train_error = evaluar(modelo, X_train, y_train)
#     test_error = evaluar(modelo, X_test, y_test)

#     return modelo, train_error, test_error

# def main(save_file=False):
#     data = np.loadtxt('TP3-ej2-conjunto.csv', delimiter=',', skiprows=1)
#     X = data[:, :-1]
#     y = np.where(data[:, -1] > 30, 1, -1)

#     # Perceptrón lineal
#     model_lineal, train_error_lineal, test_error_lineal = entrenar_perceptron_lineal(X, y)

#     # Perceptrón no lineal
#     model_nolineal, train_error_nolineal, test_error_nolineal = entrenar_perceptron_no_lineal(X, y)

#     output_data = {
#         "train_error_lineal": round(train_error_lineal, 6),
#         "test_error_lineal": round(test_error_lineal, 6),
#         "train_error_no_lineal": round(train_error_nolineal, 6),
#         "test_error_no_lineal": round(test_error_nolineal, 6)
#     }

#     if save_file:
#         timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#         os.makedirs('results', exist_ok=True)
#         root = 'results/results_ex2_' + timestamp
#         os.makedirs(root)

#         with open(f'{root}/output_ex2.json', 'w') as f:
#             json.dump(output_data, f, indent=4)

#         plt.figure(figsize=(10, 6))
#         plt.plot(model_lineal.train_errors, label='Perceptrón Lineal')
#         plt.plot(model_nolineal.train_errors, label='Perceptrón No Lineal')
#         plt.xlabel('Época')
#         plt.ylabel('Error de entrenamiento')
#         plt.title('Curva de aprendizaje')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f'{root}/learning_curves.png')

#     return output_data

# # --- Código para ejecución por terminal ---
# if __name__ == "__main__":
#     save_file = False
#     if len(sys.argv) >= 2:
#         arg = sys.argv[1].lower()
#         if arg not in ['true', 'false']:
#             print("El argumento debe ser 'true' o 'false'")
#             sys.exit(1)
#         save_file = arg == 'true'

#     print(main(save_file))
