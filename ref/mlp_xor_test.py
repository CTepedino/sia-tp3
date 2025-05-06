from ref.Perceptron import MultiLayerPerceptron


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


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

mlp = MultiLayerPerceptron(
    layers=[2, 4, 1],
    learning_rate=0.1,
    activator_function=sigmoid,
    error_function=mean_squared_error,
    weight_update_factor=sigmoid_derivative
)

mlp.train(xor_inputs, xor_outputs, 10000)

print("\n--- XOR Results ---")
for x in xor_inputs:
    out = mlp.test(x)
    print(f"Input: {x} -> Output: {out}")
