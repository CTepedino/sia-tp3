import os
import numpy as np
from PIL import Image
from Perceptron import MultiLayerPerceptron
from NonLinearFunctions import non_linear_functions

# Cargar imágenes y labels
def load_images_from_folder(folder_path, image_size=(300, 300)):
    samples = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            # Extraer el número real del nombre del archivo
            label = int(file_name.split("_")[1])
            labels.append(label)

            # Abrir imagen, pasar a escala de grises y redimensionar
            img_path = os.path.join(folder_path, file_name)
            img = Image.open(img_path).convert("L").resize(image_size)

            # Convertir imagen a array normalizado (0.0 - 1.0)
            img_array = np.array(img) / 255.0

            # Analizar si el fondo es blanco o negro
            # Se calcula el valor promedio de los píxeles
            background_value = np.mean(img_array)
            background_label = 1 if background_value > 0.5 else 0  # Fondo blanco es 0, negro es 1

            # Mostrar el valor de fondo para diagnóstico
            print(f"Background value: {background_value}, Assigned label: {background_label}")

            # Ahora aseguramos que la imagen se mantenga en 300x300 y no todo sea 1
            # Convertir la imagen a un array de 300x300 manteniendo el fondo 0 o 1
            img_array = np.where(img_array > 0.5, 1, 0)  # Asignamos 1 si es blanco (mayor que 0.5), 0 si es negro
            print(img_array)
            # Aplanar la imagen y agregarla a la lista de muestras
            flat_input = img_array.flatten().tolist()
            samples.append(flat_input)
            
            # Guardamos la etiqueta del fondo (0 o 1)
            labels.append(background_label)

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
activ_fn, activ_fn_deriv = non_linear_functions["sigmoid"]
mlp = MultiLayerPerceptron(
    layers=[100, 10],
    learning_rate=0.01,
    activator_function=activ_fn,
    error_function=lambda expected, output: sum((e - o) ** 2 for e, o in zip(expected, output)),
    weight_update_factor=activ_fn_deriv
)

# Cargar datos
train_x, train_labels = load_images_from_folder("./exer3/numeros1")
train_y = labels_to_one_hot(train_labels)

# Entrenar
mlp.train(train_x, train_y, epochs=1000)

# Testear
for i, x in enumerate(train_x):
    output = mlp.test(x)
    prediction = output.index(max(output))
    expected = train_labels[i]
    print(f"Imagen {i}: Real={expected}, Predicción={prediction} {'✅' if prediction == expected else '❌'}")
