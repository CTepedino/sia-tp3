import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generar_matriz_confusion(archivo):
    try:
        with open(archivo, 'r') as f:
            lineas = f.readlines()[1:]  # Ignorar encabezado

        matriz = np.zeros((10, 10), dtype=int)

        for linea in lineas:
            partes = linea.strip().split(',')
            if len(partes) != 2:
                continue
            esperado, predicho = partes
            try:
                esperado = int(esperado)
                predicho = int(predicho)
                if 0 <= esperado < 10 and 0 <= predicho < 10:
                    matriz[esperado][predicho] += 1
            except ValueError:
                continue

        return matriz

    except Exception as e:
        print(f"[ERROR] No se pudo procesar el archivo: {e}")
        return None

def mostrar_matriz_confusion(matriz):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicho")
    plt.ylabel("Esperado")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python matriz_confusion.py <archivo>")
    else:
        matriz = generar_matriz_confusion(sys.argv[1])
        if matriz is not None:
            mostrar_matriz_confusion(matriz)
