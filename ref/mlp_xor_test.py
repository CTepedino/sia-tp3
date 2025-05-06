from perceptron import MultiLayerPerceptron


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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
    layers=[2, 3, 1],
    learning_rate=0.1,
    activator_function=sigmoid,
    activator_derivative=sigmoid_derivative
)

mlp.train(xor_inputs, xor_outputs, 10000)

print("\n--- XOR Results ---")
for x in xor_inputs:
    out = mlp.test(x)
    print(f"Input: {x} -> Output: {out}")
