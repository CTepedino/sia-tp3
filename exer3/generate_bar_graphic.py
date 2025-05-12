import sys
import os
import matplotlib.pyplot as plt

def calcular_accuracy(path_archivo):
    with open(path_archivo, 'r') as f:
        lineas = f.readlines()[1:]  # Saltear encabezado
        total = len(lineas)
        correctos = sum(1 for l in lineas if l.strip() and l.strip().split(',')[0] == l.strip().split(',')[1])
        return correctos / total if total > 0 else 0

def procesar_directorio(directorio):
    archivos = [f for f in os.listdir(directorio) if f.endswith('.txt')]
    accuracies = [calcular_accuracy(os.path.join(directorio, archivo)) for archivo in archivos]
    return sum(accuracies) / len(accuracies) if accuracies else 0

def main():
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        print("Debe haber pares de argumentos: <directorio> <nombre>")
        return

    nombres = []
    accuracies_promedio = []

    for i in range(0, len(args), 2):
        directorio = args[i]
        nombre = args[i + 1]
        promedio = procesar_directorio(directorio)
        nombres.append(nombre)
        accuracies_promedio.append(promedio * 100)  # en porcentaje

    # Graficar
    plt.bar(nombres, accuracies_promedio, color='skyblue')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy promedio por learnig rate para relu')
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies_promedio):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig("./exer3/graphics/accuracy_por_lr_relu.png")

if __name__ == "__main__":
    main()
