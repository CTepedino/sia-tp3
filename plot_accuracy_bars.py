import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_accuracy(filename):
    correct = 0
    total = 0
    with open(filename, 'r') as f:
        # Saltar la primera línea (encabezado)
        next(f)
        # Leer el resto de las líneas
        for line in f:
            line = line.strip()
            if line:
                try:
                    # Dividir por coma y tomar ambos valores
                    pred, real = line.split(',')
                    if pred == real:
                        correct += 1
                    total += 1
                except Exception:
                    continue
    return (correct / total * 100) if total > 0 else None

def plot_accuracy_bars():
    # Configuraciones a comparar con sus colores
    configs = [
        ('leaky_relu_adam_0.0001', 'Leaky ReLU + Adam\n(lr=0.0001)', '#1f77b4'),  # Azul
        ('leaky_relu_gradient_0.0005', 'Leaky ReLU + Gradient\n(lr=0.0005)', '#ff7f0e'),  # Naranja
        ('leaky_relu_momentum_0.0005', 'Leaky ReLU + Momentum\n(lr=0.0005)', '#2ca02c')  # Verde
    ]
    
    plt.figure(figsize=(10, 6))
    
    results_dir = 'exer3/results'
    
    # Preparar datos para el gráfico
    labels = []
    means = []
    stds = []
    colors = []
    
    for config_name, label, color in configs:
        config_dir = os.path.join(results_dir, config_name)
        if os.path.exists(config_dir):
            # Buscar todos los archivos results_*.txt en el directorio
            files = glob.glob(os.path.join(config_dir, 'results_*.txt'))
            
            # Si no hay archivos results, buscar en el directorio padre
            if not files:
                files = glob.glob(os.path.join(results_dir, 'results_*.txt'))
            
            accuracies = []
            for file in files:
                try:
                    accuracy = load_accuracy(file)
                    if accuracy is not None:
                        accuracies.append(accuracy)
                except Exception:
                    continue
            
            if accuracies:
                labels.append(label)
                means.append(np.mean(accuracies))
                stds.append(np.std(accuracies))
                colors.append(color)
    
    # Crear el gráfico de barras
    x = np.arange(len(labels))
    width = 0.6
    
    bars = plt.bar(x, means, width, yerr=stds, 
                  color=colors, alpha=0.7,
                  capsize=5, ecolor='black')
    
    # Personalizar el gráfico
    plt.xlabel('Configuraciones', fontsize=12)
    plt.ylabel('Porcentaje de Acierto (%)', fontsize=12)
    plt.title('Comparación de Porcentaje de Acierto', fontsize=14, pad=20)
    plt.xticks(x, labels, rotation=0)
    
    # Agregar los valores sobre las barras
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Ajustar el rango del eje Y para mejor visualización
    plt.ylim(0, max(means) * 1.2)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_accuracy_bars() 