import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Ruta al archivo JSON de configuración (opcional)')
args = parser.parse_args()

learning_rate = 0.1
max_epochs = 10

if args.config:
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            learning_rate = config.get('learning_rate', learning_rate)
            max_epochs = config.get('max_epochs', max_epochs)
            print(f"Configuración cargada desde {args.config}")
    except Exception as e:
        print(f"No se pudo cargar el archivo de configuración: {e}")
        print("Usando valores por defecto.")

os.makedirs('results', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_results = "results/result_ex1_xor_"+timestamp
os.makedirs(save_results)

# Entradas originales
X = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

# Salidas
y = np.array([-1, 1, 1, -1])

# Peso para bias (w[0]) más peso para "cantidad de diferencias" (w[1])
w = np.random.uniform(-1, 1, 2)


def tita(x):
    return 1 if x >= 0 else -1

def cantidad_diferencias(xi):
    return int(xi[0] != xi[1])

def transformar_X(X):
    return np.array([cantidad_diferencias(xi) for xi in X])

def plot_decision_boundary(X, y, w, epoch, iteration, save_path):
    plt.clf()
    X_transf = transformar_X(X)

    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X_transf[i], 1, c='b', marker='o', label='Clase +1' if i == 0 else "")
        else:
            plt.scatter(X_transf[i], 1, c='r', marker='x', label='Clase -1' if i == 0 else "")

    if w[1] != 0:
        boundary_x = -w[0]/w[1]
        plt.plot([boundary_x, boundary_x], [0, 2], 'k--')  

    plt.xlim(-0.5, 2.5)
    plt.ylim(0.5, 1.5)
    plt.title(f'Epoch {epoch + 1}, Iteración {iteration + 1}')
    plt.xlabel('Cantidad de diferencias entre x1 y x2')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

# Entrenamiento
for epoch in range(max_epochs):
    total_error = 0
    for iteration, (xi, target) in enumerate(zip(X, y)):
        xi_transf = np.array([1, cantidad_diferencias(xi)])  # Agregamos bias
        output = tita(np.dot(w, xi_transf))
        delta = target - output
        if output != target:
            w += learning_rate * target * xi_transf

        filename = f'{save_results}/epoch{epoch + 1}_iter{iteration + 1}.png'
        plot_decision_boundary(X, y, w, epoch, iteration, filename)
        total_error+= abs(delta)
    
    if total_error == 0:
        print(f"Entrenamiento terminado en la epoca: {epoch+1}")
        break
        


# Prueba final
print("Prueba final:")
for xi in X:
    xi_transf = np.array([1, cantidad_diferencias(xi)])
    output = tita(np.dot(w, xi_transf))
    print(f"Entrada: {xi} -> Salida predicha: {output}")
