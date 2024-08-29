import numpy as np

def init_parameters(dims_layers):
    """_summary_
    This function initializes the parameters of the Neural Network based on the structure defined in the previous function.

    Args:
        dims_layers (class 'list'): list containing number of nodes per layer, incluiding input and output layers.

    Returns:
        parameters (class 'dict'): dictionary consisting of the set of initialized parameters in the network.
    """

    # Create a dictionary to store the parameters
    parameters = {}

    for num_capa in range(1, len(dims_layers)):
        # For each layer, create the corresponding weight matrix and bias.
        parameters['W' + str(num_capa)] = np.random.randn(dims_layers[num_capa], dims_layers[num_capa-1]) * (np.sqrt(2/dims_layers[num_capa-1]))
        parameters['b' + str(num_capa)] = np.zeros((dims_layers[num_capa], 1))

    # Return the initialized parameters
    return parameters