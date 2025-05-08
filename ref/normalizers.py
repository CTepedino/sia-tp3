import numpy as np

def standarize_input(inputs):
    inputs_mean = inputs.mean(axis=0)
    inputs_std = inputs.std(axis=0)
    return (inputs - inputs_mean) / inputs_std

#normaliza a [0, 1]
def normalize_output_sigmoid(outputs):
    return (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))

def denormalize_output_sigmoid(normalized_outputs, original_min, original_max):
    return normalized_outputs * (original_max - original_min) + original_min


#normaliza a [-1, 1]
def normalize_output_tanh(outputs):
    return 2 * (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs)) - 1

def denormalize_output_tanh(normalized_outputs, original_min, original_max):
    return ((normalized_outputs + 1) / 2) * (original_max - original_min) + original_min

