import math

def step(x):
    return 1 if x >= 0 else -1

def identity(x):
    return x


def hyperbolic_tangent(x):
    return math.tanh(x)

def hyperbolic_tangent_derivative(x):
    return 1 - hyperbolic_tangent(x)**2


def sigmoid(x):
    return 1 / (1 + math.exp(2*x))

def sigmoid_derivative(x):
    return 2*sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x >= 0 else 0


def softplus(x):
    return math.log(1 + math.exp(x), math.e)

def softplus_derivative(x):
    return math.exp(x)/(1+math.exp(x))


def mish(x):
    return math.tanh(softplus(x))

def mish_derivative(x):
    w = math.exp(3*x) + 4 * math.exp(2*x) + (6+4*x) * math.exp(x) + 1 + x
    d = 1+(math.exp(x)+1)**2
    return (math.exp(x)*w)/(d**2)

def leaky_relu(x, alpha=0.05):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.05):
    return 1 if x > 0 else alpha



non_linear_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (hyperbolic_tangent, hyperbolic_tangent_derivative),
    "relu": (relu, relu_derivative),
    "softplus": (softplus, softplus_derivative),
    "mish": (mish, mish_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative)
}


