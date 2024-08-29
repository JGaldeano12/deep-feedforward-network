import numpy as np

def sigmoid(z):
    """_summary_
    Implementation of the sigmoid function.

    Args:
        z (class 'numpy.ndarray'): Matrix containing the pre-activation parameters.

    Returns:
        activation (class 'numpy.ndarray'): Output matrix after applying the sigmoid function to Z.
        cache (class 'numpy.ndarray'): Returns Z.
    """
    activation = 1 / (1 + np.exp(-z))
    cache = z

    return activation, cache


def relu(z):
    """_summary_
    Implementation of the ReLU function.

    Args:
        z (class 'numpy.ndarray'): Matrix containing the pre-activation parameters.

    Returns:
        activation (class 'numpy.ndarray'): Output matrix after applying the ReLU function to Z.
        cache (class 'numpy.ndarray'): Returns Z.
    """
    activation = np.maximum(0, z)
    cache = z

    return activation, cache

def softmax(z):
    """_summary_
    Implementation of the Softmax function.

    Args:
        z (class 'numpy.ndarray'): Matrix containing the pre-activation parameters.

    Returns:
        activation (class 'numpy.ndarray'): Output matrix after applying the Softmax function to Z.
        cache (class 'numpy.ndarray'): Returns Z.
    """
    max_z = np.max(z)
    exp = np.exp(z - max_z)
    activation = exp / max_z

    return activation

def leaky_relu(z):
    """_summary_
    Implementation of the Leaky ReLU function.

    Args:
        z (class 'numpy.ndarray'): Matrix containing the pre-activation parameters.

    Returns:
        activation (class 'numpy.ndarray'): Output matrix after applying the Leaky ReLU function to Z.
        cache (class 'numpy.ndarray'): Returns Z.
    """
    activation = max(0.01 * z, z)

    return activation

def tanh(z):
    """_summary_
    Implementation of the Tanh function.

    Args:
        z (class 'numpy.ndarray'): Matrix containing the pre-activation parameters.

    Returns:
        activation (class 'numpy.ndarray'): Output matrix after applying the Tanh function to Z.
        cache (class 'numpy.ndarray'): Returns Z.
    """
    activation = np.tanh(z)

    return activation

def relu_backward(dA, cache):
    """_summary_
    Implementation of the backward propagation for the ReLU function.

    Args:
        dA (array): Gradient of the activation.
        cache (array): Cached value of Z.

    Returns:
        dZ (array): Returns the gradient of the cost with respect to Z.
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert dZ.shape == Z.shape
    
    return dZ

def sigmoid_backward(dA, cache):
    """_summary_
    Implementation of the backward propagation for the Sigmoid function.

    Args:
        dA (array): Gradient of the activation.
        cache (array): Cached value of Z.

    Returns:
        dZ (array): Returns the gradient of the cost with respect to Z.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
        
    assert dZ.shape == Z.shape
    
    return dZ