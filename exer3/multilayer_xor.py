import random
import math

# Función de activación y su derivada
def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2

# Inicializar pesos aleatoriamente
def init_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

# Producto punto entre vector y matriz
def dot(v, m):
    return [sum(v[i] * m[i][j] for i in range(len(v))) for j in range(len(m[0]))]

# Producto punto entre dos vectores
def dot_vec(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

# Entradas y salidas
X = [[-1, 1],
     [1, -1],
     [-1, -1],
     [1, 1]]
Y = [1, 1, -1, -1]

# Parámetros de la red
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

# Inicialización de pesos
w1 = init_weights(input_size, hidden_size)
b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
b2 = random.uniform(-1, 1)

# Entrenamiento
for epoch in range(epochs):
    total_error = 0
    for x, target in zip(X, Y):
        # Forward
        z1 = [dot_vec(x, [w1[i][j] for i in range(input_size)]) + b1[j] for j in range(hidden_size)]
        a1 = [tanh(z) for z in z1]

        z2 = dot_vec(a1, w2) + b2
        a2 = tanh(z2)

        # Error
        error = target - a2
        total_error += error ** 2

        # Backpropagation
        delta2 = error * tanh_derivative(z2)

        delta1 = [delta2 * w2[j] * tanh_derivative(z1[j]) for j in range(hidden_size)]

        # Actualización de pesos y sesgos
        for j in range(hidden_size):
            w2[j] += learning_rate * delta2 * a1[j]
        b2 += learning_rate * delta2

        for i in range(input_size):
            for j in range(hidden_size):
                w1[i][j] += learning_rate * delta1[j] * x[i]
        for j in range(hidden_size):
            b1[j] += learning_rate * delta1[j]

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {total_error:.4f}")

# Evaluación final
print("\nResultados:")
for x in X:
    z1 = [dot_vec(x, [w1[i][j] for i in range(input_size)]) + b1[j] for j in range(hidden_size)]
    a1 = [tanh(z) for z in z1]

    z2 = dot_vec(a1, w2) + b2
    a2 = tanh(z2)

    print(f"Entrada: {x}, Salida: {a2:.4f}, Clasificado como: {1 if a2 > 0 else -1}")
