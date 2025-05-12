import random

class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative, seed = None):
        if seed is not None:
            random.seed(seed)

        self.layers = layers
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.activator_derivative = activator_derivative