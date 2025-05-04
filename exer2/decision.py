import main as main
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    iterations = 100  # default
    if len(sys.argv) >= 2:
        number = sys.argv[1]
        if not number.isdigit() or int(number) <= 0:
            print("El argumento debe ser un número entero positivo. Se usa el default 100")
        if number.isdigit() and int(number) > 0:
            iterations = int(number)    

    executions = []
    for i in range(iterations):
        executions.append(main.main(False))

    # Calcular el promedio de los errores de test
    avg_test_lineal = np.mean([entry['test_error_lineal'] for entry in executions])
    avg_test_no_lineal = np.mean([entry['test_error_no_lineal'] for entry in executions])

    # Etiquetas para las barras
    labels = ['Error Lineal', 'Error No Lineal']
    values = [avg_test_lineal, avg_test_no_lineal]

    # Crear las barras
    bars = plt.bar(labels, values, color=['blue', 'green'])

    # Añadir etiquetas y título
    plt.ylabel('Error Promedio')
    plt.title(f'Promedio de Errores de Test en {iterations} Ejecuciones')

    # Obtener centros y alturas
    x1 = bars[0].get_x() + bars[0].get_width() / 2
    x2 = bars[1].get_x() + bars[1].get_width() / 2
    y1 = bars[0].get_height()
    y2 = bars[1].get_height()

    # Elegir una posición x intermedia para la línea vertical
    mid_x = (x1 + x2) / 2
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    # Dibujar línea vertical
    plt.vlines(mid_x, ymin, ymax, colors='black')

    tope_ancho = 0.05  # ancho de los topes
    plt.hlines(ymin, mid_x - tope_ancho, mid_x + tope_ancho, colors='black', linewidth=2)
    plt.hlines(ymax, mid_x - tope_ancho, mid_x + tope_ancho, colors='black', linewidth=2)

    # Mostrar la diferencia en la parte superior de la línea
    diff = abs(y2 - y1)
    plt.text(mid_x, ymax, f'Diff: {diff:.4f}', ha='center', va='bottom', fontsize=10, color='black')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig("promedio_errores_test.png")
    plt.show()
