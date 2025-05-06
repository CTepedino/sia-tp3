from perceptron import MultiLayerPerceptron
from activatorFunctions import non_linear_functions

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

# Cargar los datos
train_x, train_y = load_digit_data("./exer3/TP3-ej3-digitos.txt")

# Elegir funciones
activ_fn, activ_fn_deriv = non_linear_functions["tanh"]

# Instanciar y entrenar el MLP
mlp = MultiLayerPerceptron(
    [20, 1],
    0.1,
    activ_fn,
    activ_fn_deriv
)

mlp.train(train_x, train_y, epochs=1000)

# Testeo
for i in range(10):
    output = mlp.test(train_x[i])[0]
    expected = train_y[i][0]
    pred = round(output)
    print(f"Dígito {i}: Salida={output:.4f}, Predicción={"impar" if pred == 1 else "par"}, Esperado={"impar" if expected == 1 else "par"} {"✅" if pred == expected else "❌"}")
