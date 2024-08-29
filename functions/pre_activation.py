import numpy as np

def pre_activation(A, W, b):
    """_summary_
    This function calculates the pre-activation parameters of a layer.
    
    Args:
        A (class 'numpy.ndarray'): consists of the activations obtained from the previous layer.
        W (class 'numpy.ndarray'): consists of the current weights of the layer.
        b (class 'numpy.ndarray'): consists of the current bias of the layer.

    Returns:
        Z (class 'numpy.ndarray'): consists of the pre-activation params. obtained.
        pre_activation_params (class 'tuple'): tuple containing A, W, b, stored to speed up network training.
    """

    # Calculate the pre-activation parameters
    Z = np.dot(W,A) + b

    # Create a Python tuple stored in cache to speed up network training, as these parameters will be used later
    pre_activation_params = (A, W, b)

    return Z, pre_activation_params