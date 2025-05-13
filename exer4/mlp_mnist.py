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
    
    # Separar los primeros 10000 ejemplos para validación
    val_size = 10000
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]  # Quedan 50000 ejemplos
    y_train = y_train[val_size:]
    
    # Usar todo el dataset si subset_size es 50000
    if subset_size == 50000:
        print("Usando todo el conjunto de entrenamiento restante (50,000 imágenes)")
        print("Usando conjunto de validación (10,000 imágenes)")
        print("Usando conjunto de prueba (10,000 imágenes)")
    else:
        # Seleccionar elementos aleatorios del dataset restante
        indices = np.random.permutation(len(x_train))[:subset_size]
        x_train = x_train[indices]
        y_train = y_train[indices]
        print(f"Seleccionadas {subset_size} imágenes aleatorias para entrenamiento")
        print(f"Usando conjunto de validación (10,000 imágenes)")
        print(f"Usando conjunto de prueba (10,000 imágenes)")
    
    print(f"Forma de x_train: {x_train.shape}")
    print(f"Forma de x_val: {x_val.shape}")
    print(f"Forma de x_test: {x_test.shape}")
    
    # Normalizar y aplanar
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_val = x_val.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, y_train, x_val, y_val, x_test, y_test

def labels_to_one_hot(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0.0] * num_classes
        vec[label] = 1.0
        one_hot.append(vec)
    return one_hot

def guardar_estado(mlp, results_dir, precision_val):
    """Guarda los pesos y datos del optimizador solo si mejora la precisión en validación"""
    estados_dir = "./exer4/estados_entrenamiento"
    os.makedirs(estados_dir, exist_ok=True)
    
    # Verificar si existe un estado previo y su precisión
    estado_path = os.path.join(estados_dir, 'estado_entrenamiento.pkl')
    precision_anterior = 0
    
    if os.path.exists(estado_path):
        with open(estado_path, 'rb') as f:
            estado_previo = pickle.load(f)
            if 'precision_val' in estado_previo:
                precision_anterior = estado_previo['precision_val']
    
    # Solo guardar si mejora la precisión en validación
    if precision_val > precision_anterior:
        estado = {
            'weights': mlp.weights,
            'adam_data': mlp.optimizer.get_state() if hasattr(mlp.optimizer, 'get_state') else None,
            'precision_val': precision_val
        }
        with open(estado_path, 'wb') as f:
            pickle.dump(estado, f)
        print(f"Estado de entrenamiento guardado en: {estados_dir}")
        print(f"Precisión en validación anterior: {precision_anterior:.2f}% -> Nueva precisión: {precision_val:.2f}%")
    else:
        print(f"No se guardó el estado. Precisión en validación anterior: {precision_anterior:.2f}% >= Nueva precisión: {precision_val:.2f}%")

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
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_mnist()
    y_train_oh = labels_to_one_hot(y_train)
    y_val_oh = labels_to_one_hot(y_val)
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

    # 4. Evaluar en conjunto de validación
    print("\nEvaluando en el conjunto de validación:")
    correct_val = 0
    total_val = len(x_val)
    
    for i, (x, label) in enumerate(zip(x_val, y_val)):
        output = mlp.test(x)
        prediction = output.index(max(output))
        if prediction == label:
            correct_val += 1

    precision_val = (correct_val / total_val) * 100
    print(f"\nPrecisión en validación: {precision_val:.2f}%")

    # Guardar estado solo si mejora la precisión en validación
    guardar_estado(mlp, results_dir, precision_val)

    # 5. Evaluar en conjunto de prueba (solo al final)
    print("\nEvaluando en el conjunto de prueba:")
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

    precision_test = (correct_test / total_test) * 100
    print(f"\nPrecisión en prueba: {precision_test:.2f}%")

    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio_total

    # Guardar resumen final
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Precisión en validación: {precision_val:.2f}%\n")
        f.write(f"Precisión final en prueba: {precision_test:.2f}%\n")
        f.write(f"Total de imágenes correctas en prueba: {correct_test}/{total_test}\n")
        f.write(f"Error final: {mlp.error_history[-1]}\n")
        f.write(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos\n")
        f.write(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos\n")

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")

if __name__ == "__main__":
    main() 