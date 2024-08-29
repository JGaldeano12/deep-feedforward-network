import copy

def update_parameters(parameters, gradients, lrate):
    """_summary_
    Function that updates parameters using gradient descent.

    Args:
        parameters (class 'dict'): dictionary consisting of the set of initialized parameters in the network.
        gradients (class 'dict'): Python dictionary containing the gradients for each layer.
        lrate (class 'numpy.float64'): learning rate value.

    Returns:
        params (class 'dict'): dictionary consisting of the set of updated parameters in the network.
    """
    params = copy.deepcopy(parameters)
    
    # Number of hidden layers
    hidden_layers = len(params) // 2
    
    # Update the parameters across the entire network
    for num_layer in range(hidden_layers):
        params["W" + str(num_layer+1)] = params["W" + str(num_layer+1)] - lrate * gradients['dW' + str(num_layer+1)]
        params["b" + str(num_layer+1)] = params["b" + str(num_layer+1)] - lrate * gradients['db' + str(num_layer+1)]
        
    # Return the updated parameters (weights and bias)
    return params