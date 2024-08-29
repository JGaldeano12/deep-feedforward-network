import numpy as np

def linear_backward(dZ, cache):
    """_summary_
    Function to compute the backward propagation for a single layer.
    
    Args:
        dZ (class 'numpy.ndarray'): Gradient of the cost with respect to the output.
        cache (class 'tuple'): Activation values from the previous layer (A_prev), weights (W), and bias (b) of the current layer.
    
    Returns:
        dA_prev (class 'numpy.ndarray'): derivatives obtained from activations of prev. layer
        dW (class 'numpy.ndarray'): derivatives obtained from weights of actual layer.
        db (class 'numpy.ndarray'): derivatives obtained from bias of actual layer.
    """

    # Retrieve values from the cache
    A_prev, W, b = cache

    # Get the number of examples
    num_examples = A_prev.shape[1]

    # Calculate gradients with respect to W, b, and the activation layer
    dW = (1/num_examples) * np.dot(dZ, np.transpose(A_prev))
    db = (1/num_examples) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db