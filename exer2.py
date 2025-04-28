import exer2_raw
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    iterations = 100  # default
    if len(sys.argv) >= 2:
        number = sys.argv[1]
        if not number.isdigit() or int(number) <= 0:
            print("El argumento debe ser un número entero positivo.")
            sys.exit(1)
        iterations = int(number)

    executions = []
    for i in range(iterations):
        executions.append(exer2_raw.main(False))

    # Calcular el promedio de los errores de test
    avg_test_lineal = np.mean([entry['test_error_lineal'] for entry in executions])
    avg_test_no_lineal = np.mean([entry['test_error_no_lineal'] for entry in executions])

    # Etiquetas para las barras
    labels = ['Error Lineal', 'Error No Lineal']
    values = [avg_test_lineal, avg_test_no_lineal]

    # Crear las barras
    plt.bar(labels, values, color=['blue', 'green'])

    # Añadir etiquetas y título
    plt.ylabel('Error Promedio')
    plt.title('Promedio de Errores de Test')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig("promedio_errores_test.png")
    plt.show()
