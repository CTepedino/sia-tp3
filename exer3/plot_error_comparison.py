import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv

def load_training_info(filename):
    errors = []
    epochs = []
    with open(filename, 'r') as f:
        # Saltar la primera línea (encabezado)
        next(f)
        # Leer el resto de las líneas
        for line in f:
            line = line.strip()
            if line and not line.startswith('Total'):  # Ignorar líneas vacías y la línea final
                try:
                    # Dividir por coma y tomar ambos valores
                    parts = line.split(',')
                    if len(parts) >= 2:
                        epoch = int(parts[0])
                        error = float(parts[1])
                        epochs.append(epoch)
                        errors.append(error)
                except (ValueError, IndexError):
                    continue  # Ignorar líneas con formato incorrecto
    return epochs, errors

def plot_error_comparison():
    # Configuraciones a comparar con sus colores
    configs = [
        ('leaky_relu_adam_0.0001', 'Leaky ReLU + Adam (lr=0.0001)', '#1f77b4'),  # Azul
        ('leaky_relu_gradient_0.0005', 'Leaky ReLU + Gradient (lr=0.0005)', '#ff7f0e'),  # Naranja
        ('leaky_relu_momentum_0.0005', 'Leaky ReLU + Momentum (lr=0.0005)', '#2ca02c')  # Verde
    ]
    
    plt.figure(figsize=(12, 7))
    
    results_dir = 'exer3/results'
    
    # Crear elementos de leyenda personalizados
    legend_elements = [
        plt.Line2D([0], [0], color='gray', alpha=0.3, label='Ejecuciones individuales'),
        plt.Line2D([0], [0], color='gray', alpha=0.2, label='Desvío estándar')
    ]
    
    for config_name, label, color in configs:
        # Buscar todos los archivos training_info en el directorio de la configuración
        config_dir = os.path.join(results_dir, config_name)
        if os.path.exists(config_dir):
            # Buscar todos los archivos training_info_*.txt en el directorio
            files = glob.glob(os.path.join(config_dir, 'training_info_*.txt'))
            
            # Graficar cada archivo individualmente
            for file in files:
                epochs, errors = load_training_info(file)
                if errors:
                    plt.plot(epochs, errors, alpha=0.3, color='gray')
            
            # Calcular el promedio y desvío para cada época
            all_epochs = set()
            for file in files:
                epochs, _ = load_training_info(file)
                all_epochs.update(epochs)
            
            all_epochs = sorted(list(all_epochs))
            mean_errors = []
            std_errors = []
            
            for epoch in all_epochs:
                epoch_errors = []
                for file in files:
                    epochs, errors = load_training_info(file)
                    if epoch in epochs:
                        idx = epochs.index(epoch)
                        epoch_errors.append(errors[idx])
                if epoch_errors:
                    mean_errors.append(np.mean(epoch_errors))
                    std_errors.append(np.std(epoch_errors))
            
            # Graficar el promedio y desvío
            plt.plot(all_epochs, mean_errors, label=label, linewidth=2.5, color=color)
            plt.fill_between(all_epochs, 
                           np.array(mean_errors) - np.array(std_errors), 
                           np.array(mean_errors) + np.array(std_errors), 
                           alpha=0.2, color=color)
    
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Comparación de Error Promedio', fontsize=14, pad=20)
    
    # Agregar ambas leyendas
    first_legend = plt.legend(handles=legend_elements, loc='upper right', title='Elementos del gráfico')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='upper left', title='Configuraciones')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_comparison2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_error_comparison() 