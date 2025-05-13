import os
import random
from PIL import Image
from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions
import argparse
import json
from datetime import datetime



def train_test_split(X, y, labels, test_size=0.2, seed=None):
    if seed is not None:
        random.seed(seed)

    indices = list(range(len(X)))
    random.shuffle(indices)

    split_point = int(len(X) * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]

    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]

    return X_train, X_test, y_train, y_test, labels_train, labels_test

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Ruta al archivo JSON de configuración (opcional)')
parser.add_argument('--data_dir', type=str, default='./training/numeros', help='Directorio con todas las imágenes (modo original)')
parser.add_argument('--train_dir', type=str, help='Directorio con imágenes de entrenamiento (modo separado)')
parser.add_argument('--test_dir', type=str, help='Directorio con imágenes de prueba (modo separado)')
args = parser.parse_args()

learning_rate = 0.001
max_epochs = 1000
activ_fn_str = "leaky_relu"
optimizer = "gradient"  # valor por defecto

if args.config:
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            learning_rate = config.get('learning_rate', learning_rate)
            max_epochs = config.get('max_epochs', max_epochs)
            activ_fn_str = config.get('activator_function', activ_fn_str)
            optimizer = config.get('optimizer', optimizer)
            if activ_fn_str not in non_linear_functions:
                raise ValueError(f"Función de activación '{activ_fn_str}' no válida. Debe ser una de {list(non_linear_functions.keys())}.")
            if optimizer not in ["gradient", "adam", "momentum"]:
                raise ValueError(f"Optimizador '{optimizer}' no válido. Debe ser 'gradient', 'adam' o 'momentum'.")
            print(f"Configuración cargada desde {args.config}")
    except Exception as e:
        print(f"No se pudo cargar el archivo de configuración: {e}")
        print("Usando valores por defecto.")

def load_image_as_array(path_imagen, image_size=(28, 28)):
    try:
        with Image.open(path_imagen) as img:
            img = img.convert("L")  
            img = img.resize(image_size) 
            pixeles = list(img.getdata())
            normalizado = [p / 255.0 for p in pixeles]
            return normalizado
    except Exception as e:
        print(f"Error al cargar la imagen {path_imagen}: {str(e)}")
        return None

def load_images_from_folder(folder_path, image_size=(28, 28)):
    samples = []
    labels = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"El directorio {folder_path} no existe")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            try:
                # Extraer el número real del nombre del archivo
                label = int(file_name.split("_")[1])
                if label < 0 or label > 9:
                    print(f"Advertencia: Etiqueta inválida {label} en archivo {file_name}")
                    continue

                # Abrir imagen, pasar a escala de grises y redimensionar
                img_path = os.path.join(folder_path, file_name)
                img = load_image_as_array(img_path)
                if img is not None:
                    samples.append(img)
                    labels.append(label)
            except ValueError:
                print(f"Error: No se pudo extraer la etiqueta del archivo {file_name}")
                continue

    if not samples:
        raise ValueError(f"No se encontraron imágenes válidas en {folder_path}")

    return samples, labels

def labels_to_one_hot(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0.0] * num_classes
        vec[label] = 1.0
        one_hot.append(vec)
    return one_hot

def main():
    try:
        activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]
        mlp = MultiLayerPerceptron(
            [784, 30, 10],  # 784 = 28x28 píxeles
            learning_rate, 
            activ_fn,
            activ_fn_deriv,
            optimizer
        )

        if args.train_dir and args.test_dir:
            print(f"Cargando imágenes de entrenamiento desde {args.train_dir}...")
            X_train, labels_train = load_images_from_folder(args.train_dir)
            y_train = labels_to_one_hot(labels_train)

            print(f"Cargando imágenes de prueba desde {args.test_dir}...")
            X_test, labels_test = load_images_from_folder(args.test_dir)
            y_test = labels_to_one_hot(labels_test)
        else:
            print(f"Cargando imágenes desde {args.data_dir}...")
            X, labels = load_images_from_folder(args.data_dir)
            y = labels_to_one_hot(labels)
            X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
                X, y, labels, test_size=0.2, seed=42
            )

        print(f"Conjunto de entrenamiento: {len(X_train)} imágenes")
        print(f"Conjunto de prueba: {len(X_test)} imágenes")

        print("Iniciando entrenamiento...")
        mlp.train(X_train, y_train, epochs=max_epochs)

        print("\nEvaluación en conjunto de entrenamiento:")
        correct_train = 0
        for i, (x, label) in enumerate(zip(X_train, labels_train)):
            output = mlp.test(x)
            prediction = output.index(max(output))
            if prediction == label:
                correct_train += 1
            print(f"Imagen {i}: Real={label}, Predicción={prediction} {'✅' if prediction == label else '❌'}")
        
        print(f"\nPrecisión en entrenamiento: {correct_train/len(X_train)*100:.2f}%")

        dir_name = f"./exer3/results/{activ_fn_str}_{optimizer}_{learning_rate}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(dir_name, exist_ok=True)
        results_path = os.path.join(dir_name, f"results_{timestamp}.txt")

        with open(results_path, "w") as f:
            f.write("prediccion,resultado\n")  
            print("\nEvaluación en conjunto de prueba:")
            correct_test = 0
            for i, (x, label) in enumerate(zip(X_test, labels_test)):
                output = mlp.test(x)
                prediction = output.index(max(output))
                if prediction == label:
                    correct_test += 1

                print(f"Imagen {i}: Real={label}, Predicción={prediction} {'✅' if prediction == label else '❌'}")

                f.write(f"{prediction},{label}\n")
        
        print(f"\nPrecisión en prueba: {correct_test/len(X_test)*100:.2f}%")

        training_info_path = os.path.join(dir_name, f"training_info_{timestamp}.txt")
        with open(training_info_path, "w") as f:
            f.write("epoca,error_medio\n")
            for epoch, error in enumerate(mlp.error_history):
                if (epoch + 1) % 10 == 0:  
                    f.write(f"{epoch + 1},{error}\n")
            f.write(f"\nTotal de epocas recorridas: {len(mlp.error_history)}\n")

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
