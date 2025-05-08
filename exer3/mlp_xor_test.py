from perceptron import MultiLayerPerceptron
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
            if optimizer not in ["gradient", "adam"]:
                raise ValueError(f"Optimizador '{optimizer}' no válido. Debe ser 'gradient' o 'adam'.")
            print(f"Configuración cargada desde {args.config}")
    except Exception as e:
        print(f"No se pudo cargar el archivo de configuración: {e}")
        print("Usando valores por defecto.")

xor_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

xor_outputs = [
    [0],
    [1],
    [1],
    [0]
]


activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]
mlp = MultiLayerPerceptron(
    layers=[2, 3, 1],
    learning_rate=learning_rate,
    activator_function=activ_fn,
    activator_derivative=activ_fn_deriv,
    optimizer=optimizer
)

mlp.train(xor_inputs, xor_outputs, max_epochs)

print("\n--- XOR Results ---")
for i in range(len(xor_inputs)):
    out = mlp.test(xor_inputs[i])
    print(f"Input: {xor_inputs[i]} -> Output: {out}, Clasified as: {1 if out[0] >= 0.5 else 0}, Expected: {xor_outputs[i][0]}")
