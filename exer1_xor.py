import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# Entradas originales
X = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

# Salidas
y = np.array([1, -1, -1, 1])

# Peso para bias (w[0]) más peso para "cantidad de diferencias" (w[1])
w = np.random.randn(2)

learning_rate = 0.1
epochs = 10

def signo(x):
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

    x_vals = np.linspace(-0.5, 2.5, 100)
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

# Control de la convergencia para evitar cambios innecesarios
prev_w = w.copy()

# Entrenamiento
for epoch in range(epochs):
    for iteration, (xi, target) in enumerate(zip(X, y)):
        xi_transf = np.array([1, cantidad_diferencias(xi)])  # Agregamos bias
        output = signo(np.dot(w, xi_transf))
        if output != target:
            w += learning_rate * target * xi_transf
        filename = f'results/epoch{epoch + 1}_iter{iteration + 1}.png'
        plot_decision_boundary(X, y, w, epoch, iteration, filename)
        
    if np.allclose(w, prev_w, atol=1e-3): 
        print(f"Convergió en la época {epoch + 1}. No más actualizaciones.")
        break
    prev_w = w.copy()


# Prueba final
print("Prueba final:")
for xi in X:
    xi_transf = np.array([1, cantidad_diferencias(xi)])
    output = signo(np.dot(w, xi_transf))
    print(f"Entrada: {xi} -> Salida predicha: {output}")
