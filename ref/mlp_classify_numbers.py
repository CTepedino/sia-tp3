from Perceptron import MultiLayerPerceptron
from NonLinearFunctions import non_linear_functions

def load_digit_data(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    samples = []
    labels = list(range(10))  # [0, 1, ..., 9]
    
    for i in range(0, len(lines), 7):
        digit_matrix = lines[i:i+7]
        flat_input = [int(char) for row in digit_matrix for char in row.split()]
        samples.append(flat_input)

    # One-hot encoding
    outputs = []
    for label in labels:
        one_hot = [0] * 10
        one_hot[label] = 1
        outputs.append(one_hot)

    return samples, outputs

# Cargar los datos
train_x, train_y = load_digit_data("./exer3/TP3-ej3-digitos.txt")

# Elegir funciones
activ_fn, activ_fn_deriv = non_linear_functions["tanh"]

# Instanciar y entrenar el MLP
mlp = MultiLayerPerceptron(
    layers=[20, 10],  # Capa oculta de 20, salida de 10 neuronas
    learning_rate=0.1,
    activator_function=activ_fn,
    error_function=lambda expected, output: sum((e - o) ** 2 for e, o in zip(expected, output)),
    weight_update_factor=activ_fn_deriv
)

mlp.train(train_x, train_y, epochs=1000)

# Testeo
for i in range(10):
    output = mlp.test(train_x[i])
    pred = output.index(max(output))
    print(f"Dígito real: {i}, Predicción: {pred}, Salida cruda: {[round(o, 3) for o in output]} {'✅' if pred == i else '❌'}")
