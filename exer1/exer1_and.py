import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.makedirs('results', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_results = "results/result_ex1_and_"+timestamp
os.makedirs(save_results)
X = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

# Salidas
y = np.array([-1, -1, -1, 1])

# bias como w[0])
w = np.random.uniform(0,3,X.shape[1] + 1)

learning_rate = 0.1
max_epochs = 100


def tita(x):
    return 1 if x >= 0 else -1


def plot_decision_boundary(X, y, w, epoch, iteration, save_path):
    plt.clf()
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='o', label='Clase +1' if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='x', label='Clase -1' if i == 0 else "")

    x_vals = np.linspace(-2, 2, 100)

    if w[2] != 0:
        y_vals = -(w[1] * x_vals + w[0]) / w[2]
        plt.plot(x_vals, y_vals, 'k-')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f'Epoch {epoch + 1}, Iteración {iteration + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


# Entrenamiento
for epoch in range(max_epochs):
    total_error = 0
    for iteration, (xi, target) in enumerate(zip(X, y)):
        xi_ext = np.insert(xi, 0, 1)
        output = tita(np.dot(w, xi_ext))
        delta = target - output
        if output != target:
            w += learning_rate * target * xi_ext
        filename = f'{save_results}/epoch{epoch + 1}_iter{iteration + 1}.png'
        plot_decision_boundary(X, y, w, epoch, iteration, filename)
        total_error+= abs(delta)
    
    if total_error == 0:
        print(f"Entrenamiento terminado en la epoca: {epoch+1}")
        break
            


# Prueba final
print("\nPrueba final:")
for xi in X:
    xi_ext = np.insert(xi, 0, 1)
    output = tita(np.dot(w, xi_ext))
    print(f"Entrada: {xi} -> Salida predicha: {output}")