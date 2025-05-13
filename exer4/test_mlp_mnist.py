import os
import sys
import numpy as np
from tensorflow.keras.datasets import mnist
import pickle
import time

# Agregar el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from exer3.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from exer3.activatorFunctions import non_linear_functions

def cargar_mnist():
    print("Cargando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalizar y aplanar solo el conjunto de prueba
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_test, y_test

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
    # 1. Cargar datos de prueba
    x_test, y_test = cargar_mnist()
    print(f"Conjunto de prueba cargado: {len(x_test)} imágenes")

    # 2. Configurar MLP con la misma arquitectura
    learning_rate = 0.0001
    activ_fn_str = "leaky_relu"
    optimizer = "adam"
    arquitectura = [784, 128, 64, 10]

    # 3. Crear MLP y cargar pesos
    activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]
    mlp = MultiLayerPerceptron(
        arquitectura,
        learning_rate,
        activ_fn,
        activ_fn_deriv,
        optimizer
    )

    # Cargar estado guardado
    estado = cargar_estado()
    if not estado:
        print("Error: No se encontró ningún estado guardado")
        return

    mlp.weights = estado['weights']
    if estado['adam_data'] and hasattr(mlp.optimizer, 'set_state'):
        mlp.optimizer.set_state(estado['adam_data'])

    # 4. Evaluar en conjunto de prueba
    print("\nEvaluando en el conjunto de prueba...")
    correct_test = 0
    total_test = len(x_test)
    
    # Crear directorio para resultados
    results_dir = "./exer4/results/test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar resultados de predicciones
    predictions_path = os.path.join(results_dir, "test_predictions.txt")
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
            
            # Mostrar progreso cada 1000 imágenes
            if (i + 1) % 1000 == 0:
                print(f"Procesadas {i + 1}/{total_test} imágenes...")

    precision_test = (correct_test / total_test) * 100
    print(f"\nPrecisión en prueba: {precision_test:.2f}%")

    # Guardar resumen
    summary_path = os.path.join(results_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Precisión final en prueba: {precision_test:.2f}%\n")
        f.write(f"Total de imágenes correctas en prueba: {correct_test}/{total_test}\n")
        if 'precision_val' in estado:
            f.write(f"Precisión en validación del modelo guardado: {estado['precision_val']:.2f}%\n")

    print(f"\nResultados guardados en: {results_dir}")

if __name__ == "__main__":
    main() 