from datetime import datetime
import os
import copy
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative=lambda x: 1,
                 optimizer="gradient"):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_directory = "results/results_ex3_mlp_" + timestamp
        os.makedirs(results_directory, exist_ok=True)
        self.results_directory = results_directory
        self.results_file = os.path.join(results_directory, "results.txt")

        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        # Inicialización de Xavier/Glorot para los pesos
        self.weights = []

        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1 
            # Xavier/Glorot initialization: scale = sqrt(2.0 / (fan_in + fan_out))
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)).tolist())

        self.min_error = None
        self.best_weights = None
        self.patience = 50
        self.patience_counter = 0

        if self.optimizer == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = learning_rate 
            self.m = [np.zeros_like(np.array(w)) for w in self.weights] 
            self.v = [np.zeros_like(np.array(w)) for w in self.weights]  
            self.t = 0  

            if len(layers) <= 3 and max(layers) <= 10: 
                self.beta1 = 0.8  
                self.beta2 = 0.9  
                self.epsilon = 1e-6  
                self.alpha = learning_rate * 0.1  

        elif self.optimizer == "momentum":
            self.momentum = 0.9  # Factor de momentum
            self.velocity = [np.zeros_like(np.array(w)) for w in self.weights]  # Velocidad inicial

    def update_weights_gradient(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        self.weights[l] = np.array(self.weights[l]) - self.learning_rate * weight_gradients

    def update_weights_momentum(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        self.velocity[l] = self.momentum * self.velocity[l] - self.learning_rate * weight_gradients
        self.weights[l] = np.array(self.weights[l]) + self.velocity[l]

    def update_weights_adam(self, l, delta, activation):
        self.t += 1
        weight_gradients = np.outer(delta, activation)

        self.m[l] = self.beta1 * self.m[l] + (1 - self.beta1) * weight_gradients
        self.v[l] = self.beta2 * self.v[l] + (1 - self.beta2) * np.square(weight_gradients)

        m_hat = self.m[l] / (1 - self.beta1 ** self.t)
        v_hat = self.v[l] / (1 - self.beta2 ** self.t)

        update = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if len(self.layers) <= 3 and max(self.layers) <= 10:
            update = np.clip(update, -0.1, 0.1)  # Limitar actualizaciones grandes

        self.weights[l] = np.array(self.weights[l]) - update

        if self.t % 100 == 0:
            with open(self.results_file, "a") as f:
                f.write(f"ADAM Debug - Layer {l}:\n")
                f.write(f"Gradients mean: {np.mean(weight_gradients):.6f}\n")
                f.write(f"m_hat mean: {np.mean(m_hat):.6f}\n")
                f.write(f"v_hat mean: {np.mean(v_hat):.6f}\n")
                f.write(f"Update mean: {np.mean(update):.6f}\n")
                f.write(f"Weights mean: {np.mean(self.weights[l]):.6f}\n\n")

    def forward_propagation(self, input_data):
        activations = [np.array(input_data)]
        hidden_states = []

        for layer_index in range(len(self.weights)):
            prev_activation = np.append(activations[-1], 1.0)
            layer_weights = np.array(self.weights[layer_index])

            h = np.dot(layer_weights, prev_activation)
            a = np.array([self.activator_function(x) for x in h])

            hidden_states.append(h)
            activations.append(a)

        return hidden_states, activations

    def back_propagation(self, expected_output, hidden_states, activations):
        deltas = [None] * len(self.weights)
        expected_output = np.array(expected_output)

        last_layer = len(self.weights) - 1
        output_error = activations[-1] - expected_output
        deltas[last_layer] = output_error * np.array([self.activator_derivative(z) for z in hidden_states[-1]])

        for l in range(len(self.weights) - 2, -1, -1):
            next_weights = np.array(self.weights[l + 1])
            next_delta = deltas[l + 1]
            current_h = hidden_states[l]

            delta = np.dot(next_weights[:, :-1].T, next_delta) * np.array(
                [self.activator_derivative(z) for z in current_h])
            deltas[l] = delta

        for l in range(len(self.weights)):
            delta = deltas[l]
            activation = np.append(activations[l], 1.0)

            if self.optimizer == "adam":
                self.update_weights_adam(l, delta, activation)
            elif self.optimizer == "momentum":
                self.update_weights_momentum(l, delta, activation)
            else: 
                self.update_weights_gradient(l, delta, activation)

    def train(self, training_set, expected_outputs, epochs):
        best_error = float('inf')
        error_history = []
        min_delta = 1e-5
        window_size = 10

        # Inicializar ADAM si es necesario
        if self.optimizer == "adam":
            self.t = 0
            self.m = [np.zeros_like(np.array(w)) for w in self.weights]
            self.v = [np.zeros_like(np.array(w)) for w in self.weights]

        for epoch in range(epochs):
            error = 0
            np.random.seed(epoch)
            indices = np.random.permutation(len(training_set))

            for idx in indices:
                x = training_set[idx]
                y = expected_outputs[idx]

                hidden_states, activations = self.forward_propagation(x)
                output = activations[-1]
                self.back_propagation(y, hidden_states, activations)

                error += 0.5 * np.sum((np.array(y) - output) ** 2)

            average_error = error / len(training_set)
            error_history.append(average_error)

            if average_error < (best_error - min_delta):
                best_error = average_error
                self.best_weights = copy.deepcopy(self.weights)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

                if len(error_history) >= window_size:
                    recent_errors = error_history[-window_size:]
                    if all(recent_errors[i] > recent_errors[i - 1] * 1.01 for i in range(1, len(recent_errors))):
                        print(
                            f"Early stopping en época {epoch + 1} - Error aumenta consistentemente en {window_size} épocas")
                        break

                if self.patience_counter >= self.patience:
                    print(
                        f"Early stopping en época {epoch + 1} - Paciencia agotada después de {self.patience} épocas sin mejora")
                    break

            with open(self.results_file, "a") as f:
                log_line = f"epoch {epoch + 1} average error - {average_error}\n"
                f.write(log_line)
                if (epoch + 1) % 10 == 0:
                    print(log_line.strip())
                    if self.optimizer == "adam":
                        print(f"ADAM - t: {self.t}, learning rate: {self.alpha}")

        self.weights = copy.deepcopy(self.best_weights)
        self.error_history = error_history  # Guardar el historial de errores

    def test(self, input_data):
        hidden_states, activations = self.forward_propagation(input_data)
        return activations[-1].tolist()
