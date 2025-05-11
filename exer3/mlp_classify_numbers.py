from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Ruta al archivo JSON de configuración (opcional)')
args = parser.parse_args()

learning_rate = 0.1
max_epochs = 1000
activ_fn_str = "tanh"
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
train_x, train_y = load_digit_data("./training/TP3-ej3-digitos.txt")

# Elegir funciones
activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]

# Instanciar y entrenar el MLP
mlp = MultiLayerPerceptron(
    [35, 10],  # Capa oculta de 20, salida de 10 neuronas
    learning_rate,
    activ_fn,
    activ_fn_deriv,
    optimizer
)

mlp.train(train_x, train_y, epochs=max_epochs)

# Testeo
for i in range(10):
    output = mlp.test(train_x[i])
    pred = output.index(max(output))
    print(f"Dígito real: {i}, Predicción: {pred}, Salida cruda: {[round(o, 3) for o in output]} {'✅' if pred == i else '❌'}")
