import os
import sys
import numpy as np
from tensorflow.keras.datasets import mnist
import pickle
import time
from datetime import datetime

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from exer3.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from exer3.activatorFunctions import non_linear_functions

def load_mnist(subset_size=1000):
    print("Cargando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    
    val_size = 10000
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]  
    y_train = y_train[val_size:]
    
    
    if subset_size == 50000:
        print("Usando todo el conjunto de entrenamiento restante (50,000 imágenes)")
        print("Usando conjunto de validación (10,000 imágenes)")
        print("Usando conjunto de prueba (10,000 imágenes)")
    else:
      
        indices = np.random.permutation(len(x_train))[:subset_size]
        x_train = x_train[indices]
        y_train = y_train[indices]
        print(f"Seleccionadas {subset_size} imágenes aleatorias para entrenamiento")
        print(f"Usando conjunto de validación (10,000 imágenes)")
        print(f"Usando conjunto de prueba (10,000 imágenes)")
    
    print(f"Forma de x_train: {x_train.shape}")
    print(f"Forma de x_val: {x_val.shape}")
    print(f"Forma de x_test: {x_test.shape}")
    
    
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

def save_state(mlp, results_dir, val_accuracy):
    """Guarda los pesos y datos del optimizador solo si mejora la precisión en validación"""
    states_dir = "./exer4/estados_entrenamiento"
    os.makedirs(states_dir, exist_ok=True)
    
    
    state_path = os.path.join(states_dir, 'estado_entrenamiento.pkl')
    prev_accuracy = 0
    
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            prev_state = pickle.load(f)
            if 'precision_val' in prev_state:
                prev_accuracy = prev_state['precision_val']
    
    
    if val_accuracy > prev_accuracy:
        state = {
            'weights': mlp.weights,
            'adam_data': mlp.optimizer.get_state() if hasattr(mlp.optimizer, 'get_state') else None,
            'precision_val': val_accuracy
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Estado de entrenamiento guardado en: {states_dir}")
        print(f"Precisión en validación anterior: {prev_accuracy:.2f}% -> Nueva precisión: {val_accuracy:.2f}%")
    else:
        print(f"No se guardó el estado. Precisión en validación anterior: {prev_accuracy:.2f}% >= Nueva precisión: {val_accuracy:.2f}%")

def load_state():
    """Carga los pesos y datos del optimizador si existen"""
    states_dir = "./exer4/estados_entrenamiento"
    state_path = os.path.join(states_dir, 'estado_entrenamiento.pkl')
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        print("Estado de entrenamiento cargado desde:", states_dir)
        return state
    return None

def main():
    
    total_start_time = time.time()

    
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    y_train_oh = labels_to_one_hot(y_train)
    y_val_oh = labels_to_one_hot(y_val)
    y_test_oh = labels_to_one_hot(y_test)

    
    learning_rate = 0.00005
    max_epochs = 5
    activ_fn_str = "leaky_relu"
    optimizer = "adam"
    architecture = [784, 128, 64, 10]  

    
    results_dir = f"./exer4/results/mnist_results"
    os.makedirs(results_dir, exist_ok=True)

    
    config_path = os.path.join(results_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Configuración del entrenamiento:\n")
        f.write(f"Arquitectura: {architecture}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Épocas: {max_epochs}\n")
        f.write(f"Función de activación: {activ_fn_str}\n")
        f.write(f"Optimizador: {optimizer}\n")
        f.write(f"Tamaño del dataset: {len(x_train)} imágenes de entrenamiento\n")
        f.write(f"Tamaño de prueba: {len(x_test)} imágenes de prueba\n")

    print("\nConfigurando MLP...")
    print(f"Arquitectura: {architecture}")
    print(f"Learning rate: {learning_rate}")
    print(f"Épocas: {max_epochs}")
    print(f"Función de activación: {activ_fn_str}")
    print(f"Optimizador: {optimizer}")

    activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]
    mlp = MultiLayerPerceptron(
        architecture,
        learning_rate,
        activ_fn,
        activ_fn_deriv,
        optimizer
    )

    
    prev_state = load_state()
    if prev_state:
        mlp.weights = prev_state['weights']
        if prev_state['adam_data'] and hasattr(mlp.optimizer, 'set_state'):
            mlp.optimizer.set_state(prev_state['adam_data'])

   
    print("\nEntrenando el MLP sobre MNIST...")
    training_start_time = time.time()
    mlp.train(x_train, y_train_oh, epochs=max_epochs)
    training_time = time.time() - training_start_time

    
    print("\nEvaluando en el conjunto de validación:")
    correct_val = 0
    total_val = len(x_val)
    
    for i, (x, label) in enumerate(zip(x_val, y_val)):
        output = mlp.test(x)
        prediction = output.index(max(output))
        if prediction == label:
            correct_val += 1

    val_accuracy = (correct_val / total_val) * 100
    print(f"\nPrecisión en validación: {val_accuracy:.2f}%")

    
    save_state(mlp, results_dir, val_accuracy)

    
    print("\nEvaluando en el conjunto de prueba:")
    correct_test = 0
    total_test = len(x_test)
    
    
    predictions_path = os.path.join(results_dir, "predictions.txt")
    with open(predictions_path, "w") as f:
        f.write("imagen,real,prediccion,correcto\n")
        
        for i, (x, label) in enumerate(zip(x_test, y_test)):
            output = mlp.test(x)
            prediction = output.index(max(output))
            is_correct = prediction == label
            if is_correct:
                correct_test += 1
            
            
            f.write(f"{i},{label},{prediction},{1 if is_correct else 0}\n")
            
            
            if (i + 1) % 200 == 0:
                print(f"Procesadas {i + 1}/{total_test} imágenes...")

    test_accuracy = (correct_test / total_test) * 100
    print(f"\nPrecisión en prueba: {test_accuracy:.2f}%")

    
    total_time = time.time() - total_start_time

    
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Precisión en validación: {val_accuracy:.2f}%\n")
        f.write(f"Precisión final en prueba: {test_accuracy:.2f}%\n")
        f.write(f"Total de imágenes correctas en prueba: {correct_test}/{total_test}\n")
        f.write(f"Error final: {mlp.error_history[-1]}\n")
        f.write(f"Tiempo de entrenamiento: {training_time:.2f} segundos\n")
        f.write(f"Tiempo total de ejecución: {total_time:.2f} segundos\n")

    print(f"\nResultados guardados en: {results_dir}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"Tiempo total: {total_time:.2f} segundos")

if __name__ == "__main__":
    main() 