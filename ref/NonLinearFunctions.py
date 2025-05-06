import math

def hiperbolic_tangent(x, b = 1):
    return math.tanh(b*x)

def hiperbolic_tangent_derivative(x, b = 1):
    return b * (1 - hiperbolic_tangent(x, b)**2)


def sigmoid(x, b = 1):
    return 1 / (1 + math.exp(2*b*x))

def sigmoid_derivative(x, b = 1):
    return 2*b*sigmoid(x, b)*(1-sigmoid(x, b))




