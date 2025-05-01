import exer2.main as main
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
import os

def probar_diferentes_lr(X, y, lrs, epochs=100, repeticiones=10):
    resultados = []

    for lr in lrs:
        test_errors_l = []
        test_errors_nl = []

        for _ in range(repeticiones):
            _, _, test_error_l = main.entrenar_lineal(X, y, lr=lr, epochs=epochs)
            _, _, test_error_nl = main.entrenar_no_lineal(X, y, lr=lr, epochs=epochs)

            test_errors_l.append(test_error_l)
            test_errors_nl.append(test_error_nl)

        resultados.append({
            "lr": lr,
            "test_error_lineal_prom": np.mean(test_errors_l),
            "test_error_lineal_std": np.std(test_errors_l),
            "test_error_no_lineal_prom": np.mean(test_errors_nl),
            "test_error_no_lineal_std": np.std(test_errors_nl)
        })

    return resultados

data = np.loadtxt('TP3-ej2-escalado.csv', delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# lrs = [0.0001, 0.0005,0.001,0.005, 0.0075, 0.01, 0.0125]
lrs = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125]
# Ejecutar
repeticiones=500
resultados = probar_diferentes_lr(X, y, lrs, epochs=500, repeticiones=repeticiones)

# Extraer info
lrs = [res["lr"] for res in resultados]
test_error_lineal_prom = [res["test_error_lineal_prom"] for res in resultados]
test_error_lineal_std = [res["test_error_lineal_std"] for res in resultados]
test_error_nolineal_prom = [res["test_error_no_lineal_prom"] for res in resultados]
test_error_nolineal_std = [res["test_error_no_lineal_std"] for res in resultados]

# ... (todo igual hasta antes de mostrar el gráfico)

plt.figure(figsize=(10, 6))
plt.errorbar(
    lrs,
    test_error_lineal_prom,
    yerr=test_error_lineal_std,
    fmt='o-',
    label='Lineal',
    capsize=5
)

plt.errorbar(
    lrs,
    test_error_nolineal_prom,
    yerr=test_error_nolineal_std,
    fmt='s-',
    label='No Lineal',
    capsize=5
)

# Encontrar mínimos y marcarlos en rojo
idx_min_lineal = np.argmin(test_error_lineal_prom)
idx_min_nolineal = np.argmin(test_error_nolineal_prom)

plt.scatter(
    lrs[idx_min_lineal],
    test_error_lineal_prom[idx_min_lineal],
    color='red',
    s=100,
    zorder=5,
    label='Mínimo Lineal'
)

plt.scatter(
    lrs[idx_min_nolineal],
    test_error_nolineal_prom[idx_min_nolineal],
    color='red',
    s=100,
    zorder=5,
    marker='D',
    label='Mínimo No Lineal'
)

plt.xlabel('Learning Rate (lr)')
plt.ylabel('Error de test (promedio ± std)')
plt.title(f'Error de test promedio vs Learning Rate en {repeticiones} repeticiones')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
root = f'results/'
os.makedirs(root, exist_ok=True)
plt.savefig(f'results/lr_vs_test_error_{timestamp}.png')
