import numpy as np
from activation_functions import relu, sigmoid
from pre_activation import pre_activation

def forward_activation(dataset, parameters):
    """_summary_
    Function that computes the activation of a layer.

    Args:
        dataset (class 'numpy.ndarray'): Input data. Used to determine the number of neurons in the input layer.
        parameters (class 'dict'): Python dictionary with the initialized parameters based on the network structure: weights and biases.

    Returns:
        AL (class 'numpy.ndarray'): Activation of the final layer.
        cache (class 'list'): Tuple containing the activation of the previous layer (A_prev), weights of the current layer (W), and bias (b).
    """

    # Store the results in cache to speed up network training
    caches = []

    # Number of hidden layers
    hidden_layers = len(parameters) // 2

    # For the first layer, activation functions are the input data
    A = dataset

    # Compute activation for hidden layers
    for num_layer in range(1, hidden_layers):
        Z, pre_activation_params = pre_activation(A, parameters['W' + str(num_layer)], parameters['b' + str(num_layer)])
        A, cache = relu(Z)
        cache = (pre_activation_params, cache)
        caches.append(cache)
    
    # Compute activation for the final layer
    Z, pre_activation_params = pre_activation(A, parameters['W' + str(hidden_layers)], parameters['b' + str(hidden_layers)])
    AL, cache = sigmoid(Z)
    cache = (pre_activation_params, cache)
    caches.append(cache)
    
    return AL, caches