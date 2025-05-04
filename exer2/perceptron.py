import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.01, epochs=100):
        self.w = np.random.randn(input_dim + 1)
        self.lr = lr
        self.epochs = epochs
        self.train_errors = []

    def predict(self, x):
        x = np.insert(x, 0, 1)
        return np.dot(self.w, x)

    def fit(self, X, y):
        for _ in range(self.epochs):
            errores = 0
            for xi, target in zip(X, y):
                xi_bias = np.insert(xi, 0, 1)
                pred = np.dot(self.w, xi_bias)
                error = pred - target
                self.w -= self.lr * error * xi_bias
                errores += error**2
            self.train_errors.append(errores / len(y))
