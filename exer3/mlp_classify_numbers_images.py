import os
import numpy as np
from PIL import Image
from perceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions
from sklearn.model_selection import train_test_split

# Cargar imágenes y labels
def load_image_as_array(path_imagen, image_size=(28, 28)):
    try:
        with Image.open(path_imagen) as img:
            img = img.convert("L")  # escala de grises
            img = img.resize(image_size)  # asegurate que todos tengan la misma forma
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

# Convertir a vectores one-hot
def labels_to_one_hot(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0.0] * num_classes
        vec[label] = 1.0
        one_hot.append(vec)
    return one_hot

def main():
    try:
        # Configurar MLP
        activ_fn, activ_fn_deriv = non_linear_functions["leaky_relu"]
        mlp = MultiLayerPerceptron(
            [784, 15, 10],  # 784 = 28x28 píxeles
            0.01, 
            activ_fn,
            activ_fn_deriv
        )

        # Cargar datos
        data_path = "./training/numeros"
        print(f"Cargando imágenes desde {data_path}...")
        X, labels = load_images_from_folder(data_path)
        y = labels_to_one_hot(labels)

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X, y, labels, test_size=0.2, random_state=42
        )

        print(f"Conjunto de entrenamiento: {len(X_train)} imágenes")
        print(f"Conjunto de prueba: {len(X_test)} imágenes")

        # Entrenar
        print("Iniciando entrenamiento...")
        mlp.train(X_train, y_train, epochs=1000)

        # Evaluar en conjunto de entrenamiento
        print("\nEvaluación en conjunto de entrenamiento:")
        correct_train = 0
        for i, (x, label) in enumerate(zip(X_train, labels_train)):
            output = mlp.test(x)
            prediction = output.index(max(output))
            if prediction == label:
                correct_train += 1
            print(f"Imagen {i}: Real={label}, Predicción={prediction} {'✅' if prediction == label else '❌'}")
        
        print(f"\nPrecisión en entrenamiento: {correct_train/len(X_train)*100:.2f}%")

        # Evaluar en conjunto de prueba
        print("\nEvaluación en conjunto de prueba:")
        correct_test = 0
        for i, (x, label) in enumerate(zip(X_test, labels_test)):
            output = mlp.test(x)
            prediction = output.index(max(output))
            if prediction == label:
                correct_test += 1
            print(f"Imagen {i}: Real={label}, Predicción={prediction} {'✅' if prediction == label else '❌'}")
        
        print(f"\nPrecisión en prueba: {correct_test/len(X_test)*100:.2f}%")

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
