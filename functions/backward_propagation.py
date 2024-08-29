import numpy as np
from activation_functions import sigmoid_backward, relu_backward
from linear_backward import linear_backward

def backward_propagation(dZ, Y, cache):
    """_summary_
    Function that implements 'backward propagation' for a single layer.

    Args:
        dZ (class 'numpy.ndarray'): Gradient of the cost with respect to the output.
        Y (class 'numpy.ndarray'): true labels.
        cache (class 'list'): Tuple containing the activation of the previous layer (A_prev), weights of the current layer (W), and bias (b).

    Return:
        gradients (class 'dict'): Python dictionary containing the gradients for each layer.
    """
    # Initialize the gradient vector
    gradients = {}

    # The number of layers can be determined directly from the number of caches
    num_layers = len(cache)

    Y = Y.reshape(dZ.shape)
    
    # To avoid logarithms and divisions by 0
    epsilon = 1e-15
    AL = np.clip(dZ, epsilon, 1 - epsilon)

    # Initialize backward propagation by calculating the derivative of the final activation layer
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Start with the final layer, from which parameters can be obtained using the 'linear_activation_backward' function
    cache_actual = cache[num_layers-1]
    linear_cache, activation_cache = cache_actual
    dZ = sigmoid_backward(dAL, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    gradients['dA'+str(num_layers-1)] = dA_prev
    gradients['dW'+str(num_layers)] = dW
    gradients['db'+str(num_layers)] = db

    # Iterate from the last layer to the first, i.e., in reverse order
    for num_layer in reversed(range(num_layers-1)):
        # Retrieve the cache for the layer
        cache_actual = cache[num_layer]
        linear_cache, activation_cache = cache_actual

        # Calculate the backward linear activation
        dZ = relu_backward(gradients['dA'+str(num_layer+1)], activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        gradients['dA'+str(num_layer)] = dA_prev
        gradients['dW'+str(num_layer+1)] = dW
        gradients['db'+str(num_layer+1)] = db

    return gradients