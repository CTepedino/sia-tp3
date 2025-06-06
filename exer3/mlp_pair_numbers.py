from perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions
import argparse
import json
import os

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

    outputs = [[label % 2] for label in labels]

    return samples, outputs

train_x, train_y = load_digit_data("./training/TP3-ej3-digitos.txt")

activ_fn, activ_fn_deriv = non_linear_functions[activ_fn_str]

mlp = MultiLayerPerceptron(
    #35 x el tamaño de la entrada
    [35, 1],
    learning_rate,
    activ_fn,
    activ_fn_deriv,
    optimizer
)

mlp.train(train_x, train_y, epochs=max_epochs)

# Crear el directorio si no existe
os.makedirs('./exer3/results', exist_ok=True)

# Ruta del archivo de resultados
result_file_path = './exer3/results/resultados_ej3b.txt'

# Escribir el header solo si el archivo no existe
if not os.path.exists(result_file_path):
    with open(result_file_path, 'w') as f:
        f.write("prediccion,resultado\n")

# Append de resultados
with open(result_file_path, 'a') as f:
    for i in range(10):
        output = mlp.test(train_x[i])[0]
        expected = train_y[i][0]
        pred = round(output)
        resultado_str = '✅' if pred == expected else '❌'
        f.write(f"{'impar' if pred == 1 else 'par'},{'impar' if expected == 1 else 'par'}\n")
        print(f"Dígito {i}: Salida={output:.4f}, Predicción={'impar' if pred == 1 else 'par'}, Esperado={'impar' if expected == 1 else 'par'} {resultado_str}")
