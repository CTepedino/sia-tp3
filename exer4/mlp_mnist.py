import os
import sys
import numpy as np
from tensorflow.keras.datasets import mnist
import pickle
import time
from datetime import datetime
# Agregar el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from exer3.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from exer3.activatorFunctions import non_linear_functions

def cargar_mnist(subset_size=3000):
    print("Cargando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Usar todo el dataset si subset_size es 60000
    if subset_size == 60000:
        print("Usando todo el conjunto de entrenamiento (60,000 imágenes)")
        print("Usando todo el conjunto de prueba (10,000 imágenes)")
    else:
        # Seleccionar elementos aleatorios del dataset
        indices = np.random.permutation(len(x_train))[:subset_size]
        x_train = x_train[indices]
        y_train = y_train[indices]
        print(f"Seleccionadas {subset_size} imágenes aleatorias para entrenamiento")
    
    print(f"Forma de x_train: {x_train.shape}")
    print(f"Forma de x_test: {x_test.shape}")
    
    # Normalizar y aplanar
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, y_train, x_test, y_test

def labels_to_one_hot(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0.0] * num_classes
        vec[label] = 1.0
        one_hot.append(vec)
    return one_hot

def guardar_estado(mlp, results_dir, precision_actual):
    """Guarda los pesos y datos del optimizador solo si mejora la precisión"""
    estados_dir = "./exer4/estados_entrenamiento"
    os.makedirs(estados_dir, exist_ok=True)
    
    # Verificar si existe un estado previo y su precisión
    estado_path = os.path.join(estados_dir, 'estado_entrenamiento.pkl')
    precision_anterior = 0
    
    if os.path.exists(estado_path):
        with open(estado_path, 'rb') as f:
            estado_previo = pickle.load(f)
            if 'precision' in estado_previo:
                precision_anterior = estado_previo['precision']
    
    # Solo guardar si mejora la precisión
    if precision_actual > precision_anterior:
        estado = {
            'weights': mlp.weights,
            'adam_data': mlp.optimizer.get_state() if hasattr(mlp.optimizer, 'get_state') else None,
            'precision': precision_actual
        }
        with open(estado_path, 'wb') as f:
            pickle.dump(estado, f)
        print(f"Estado de entrenamiento guardado en: {estados_dir}")
        print(f"Precisión anterior: {precision_anterior:.2f}% -> Nueva precisión: {precision_actual:.2f}%")
    else:
        print(f"No se guardó el estado. Precisión anterior: {precision_anterior:.2f}% >= Nueva precisión: {precision_actual:.2f}%")

def cargar_estado():
    """Carga los pesos y datos del optimizador si existen"""
    estados_dir = "./exer4/estados_entrenamiento"
    estado_path = os.path.join(estados_dir, 'estado_entrenamiento.pkl')
    if os.path.exists(estado_path):
        with open(estado_path, 'rb') as f:
            estado = pickle.load(f)
        print("Estado de entrenamiento cargado desde:", estados_dir)
        return estado
    return None

def main():
    # Iniciar medición de tiempo total
    tiempo_inicio_total = time.time()

    # 1. Cargar y preparar datos (usando todo el dataset)
    x_train, y_train, x_test, y_test = cargar_mnist()
    y_train_oh = labels_to_one_hot(y_train)
    y_test_oh = labels_to_one_hot(y_test)

    # 2. Configurar MLP con hiperparámetros recomendados
    learning_rate = 0.0001
    max_epochs = 30
    activ_fn_str = "leaky_relu"
    optimizer = "adam"
    arquitectura = [784, 128, 64, 10]  # Arquitectura más compacta

    # Crear directorio para resultados si no existe
    results_dir = f"./exer4/results/mnist_results"
    os.makedirs(results_dir, exist_ok=True)

    # Guardar configuración
    config_path = os.path.join(results_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Configuracion del entrenamiento:\n")
        f.write(f"Arquitectura: {arquitectura}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epocas: {max_epochs}\n")
        f.write(f"Funcion de activación: {activ_fn_str}\n")
        f.write(f"Optimizador: {optimizer}\n")
        f.write(f"Dataset size: {len(x_train)} imágenes de entrenamiento\n")
        f.write(f"Test size: {len(x_test)} imágenes de prueba\n")

    print("\nConfigurando MLP...")
    print(f"Arquitectura: {arquitectura}")
    print(f"Learning rate: {learning_rate}")
    print(f"Épocas: {max_epochs}")
    print(f"Función de activación: {activ_fn_str}")
    print(f"Optimizador: {optimizer}")

    activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]
    mlp = MultiLayerPerceptron(
        arquitectura,
        learning_rate,
        activ_fn,
        activ_fn_deriv,
        optimizer
    )

    # Intentar cargar estado previo si existe
    estado_previo = cargar_estado()
    if estado_previo:
        mlp.weights = estado_previo['weights']
        if estado_previo['adam_data'] and hasattr(mlp.optimizer, 'set_state'):
            mlp.optimizer.set_state(estado_previo['adam_data'])

    # 3. Entrenar
    print("\nEntrenando el MLP sobre MNIST...")
    tiempo_inicio_entrenamiento = time.time()
    mlp.train(x_train, y_train_oh, epochs=max_epochs)
    tiempo_entrenamiento = time.time() - tiempo_inicio_entrenamiento

    # 4. Evaluar
    print("\nEvaluando en el set de prueba:")
    correct_test = 0
    total_test = len(x_test)
    
    # Guardar resultados de predicciones
    predictions_path = os.path.join(results_dir, "predictions.txt")
    with open(predictions_path, "w") as f:
        f.write("imagen,real,prediccion,correcto\n")
        
        for i, (x, label) in enumerate(zip(x_test, y_test)):
            output = mlp.test(x)
            prediction = output.index(max(output))
            is_correct = prediction == label
            if is_correct:
                correct_test += 1
            
            # Guardar predicción
            f.write(f"{i},{label},{prediction},{1 if is_correct else 0}\n")
            
            # Mostrar progreso cada 200 imágenes
            if (i + 1) % 200 == 0:
                print(f"Procesadas {i + 1}/{total_test} imágenes...")

    precision = (correct_test / total_test) * 100
    print(f"\nPrecisión en prueba: {precision:.2f}%")

    # Guardar estado solo si mejora la precisión
    guardar_estado(mlp, results_dir, precision)

    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio_total

    # Guardar resumen final
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Precisión final en prueba: {precision:.2f}%\n")
        f.write(f"Total de imágenes correctas: {correct_test}/{total_test}\n")
        f.write(f"Error final: {mlp.error_history[-1]}\n")
        f.write(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos\n")
        f.write(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos\n")

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")

if __name__ == "__main__":
    main() 