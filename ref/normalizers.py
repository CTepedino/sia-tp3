import numpy as np

def standarize_input(inputs):
    inputs_mean = inputs.mean(axis=0)
    inputs_std = inputs.std(axis=0)
    return (inputs - inputs_mean) / inputs_std

#normaliza a [0, 1]
def normalize_output_sigmoid(outputs):
    min_val = np.min(outputs)
    max_val = np.max(outputs)
    normalized = (outputs - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def denormalize_output_sigmoid(normalized_output, original_min, original_max):
    return normalized_output * (original_max - original_min) + original_min


#normaliza a [-1, 1]
#normaliza a [0, 1]
def normalize_output_tanh(outputs):
    min_val = np.min(outputs)
    max_val = np.max(outputs)
    normalized = 2 * (outputs - min_val) / (max_val - min_val) - 1
    return normalized, min_val, max_val

def denormalize_output_tanh(normalized_output, original_min, original_max):
    return ((normalized_output + 1) / 2) * (original_max - original_min) + original_min

