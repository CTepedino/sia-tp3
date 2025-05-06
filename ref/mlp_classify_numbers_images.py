import os
import numpy as np
from PIL import Image
from perceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions

# Cargar imágenes y labels
def load_image_as_array(path_imagen, image_size=(28, 28)):
    with Image.open(path_imagen) as img:
        img = img.convert("L")  # escala de grises
        img = img.resize(image_size)  # asegurate que todos tengan la misma forma
        pixeles = list(img.getdata())
        normalizado = [p / 255.0 for p in pixeles]
        return normalizado



def load_images_from_folder(folder_path, image_size=(28, 28)):
    samples = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            # Extraer el número real del nombre del archivo
            label = int(file_name.split("_")[1])
            labels.append(label)

            # Abrir imagen, pasar a escala de grises y redimensionar
            img_path = os.path.join(folder_path, file_name)
            img = load_image_as_array(img_path)
            samples.append(img)

    return samples, labels

# Convertir a vectores one-hot
def labels_to_one_hot(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0.0] * num_classes
        vec[label] = 1.0
        one_hot.append(vec)
    return one_hot

# Configurar MLP
activ_fn, activ_fn_deriv = non_linear_functions["leaky_relu"]
mlp = MultiLayerPerceptron(
    [128, 64, 10],  
    0.01, 
    activ_fn,
    activ_fn_deriv
)

# Cargar datos
train_x, train_labels = load_images_from_folder("./exer3/numeros")
train_y = labels_to_one_hot(train_labels)

# Entrenar
mlp.train(train_x, train_y, epochs=1000)

# Testear
for i, x in enumerate(train_x):
    output = mlp.test(x)
    prediction = output.index(max(output))
    expected = train_labels[i]
    print(f"Imagen {i}: Real={expected}, Predicción={prediction} {'✅' if prediction == expected else '❌'}")
