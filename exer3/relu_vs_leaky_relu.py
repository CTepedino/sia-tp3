import sys
import os
import matplotlib.pyplot as plt

import re

def extract_accuracy_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            total = 0
            aciertos = 0

            for line in lines:
                partes = line.strip().split(',')
                if len(partes) == 2:
                    esperado, predicho = partes
                    total += 1
                    if esperado == predicho:
                        aciertos += 1

            if total == 0:
                return 0.0

            accuracy = (aciertos / total) * 100  # porcentaje
            return accuracy

    except Exception as e:
        print(f"[ERROR] No se pudo leer {file_path}: {e}")
        return None



def extract_lr_from_folder_name(folder_name):
    try:
        # print(f"Extracting LR from folder: {folder_name}")
        return float(folder_name.split("_")[-1].replace("e", "E"))  # Soporta notación científica
        
    except:
        return None

def main(directories):
    relu_data = []
    leaky_relu_data = []

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            print(f"[WARN] {dir_path} no es un directorio válido.")
            continue

        files = sorted(os.listdir(dir_path))
        if not files:
            print(f"[WARN] No hay archivos en {dir_path}")
            continue

        first_file_path = os.path.join(dir_path, files[0])
        
        accuracy = extract_accuracy_from_file(first_file_path)
        
        lr = extract_lr_from_folder_name(os.path.basename(dir_path.rstrip("/\\")))

        
        if accuracy is None or lr is None:
            print(f"accuracy {accuracy} y lr {lr} inválidos en {dir_path}")
            # print(f"[WARN] Accuracy o LR inválido en {dir_path}")
            continue

        if "leaky_relu" in dir_path:
            leaky_relu_data.append((lr, accuracy))
        elif "relu" in dir_path:
            relu_data.append((lr, accuracy))

    # Ordenar por learning rate
    relu_data.sort()
    leaky_relu_data.sort()

    # Separar para graficar
    relu_lrs, relu_accs = zip(*relu_data) if relu_data else ([], [])
    leaky_lrs, leaky_accs = zip(*leaky_relu_data) if leaky_relu_data else ([], [])

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(relu_lrs, relu_accs, label='ReLU', marker='o')
    plt.plot(leaky_lrs, leaky_accs, label='Leaky ReLU', marker='o')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparación de accuracy según función de activación')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('./exer3/graphics/accuracy_por_lr_relu_vs_leaky_relu.png')
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python generate_relu_vs_leaky_graphic.py <dir1> <dir2> ...")
    else:
        main(sys.argv[1:])
