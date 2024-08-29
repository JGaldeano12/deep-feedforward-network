def init_layers(X, Y, num_hidden_layers, num_hidden_units):
    """_summary_
    This method defines the structure of the neural network by specifying the number of neurons in the input layer (which is determined by the size of the dataset used),
    the number of hidden layers specified by an input parameter, the number of neurons in each of these hidden layers, and finally, the number of neurons in the output layer 
    (determined by the size of the target variable).

    It is understood that this is building a Deep Neural Network where all hidden layers have the same number of neurons.

    Args:
        X (class 'numpy.ndarray'): input dataset (input size, number of examples)
        Y (class 'numpy.ndarray'): labels (output size, number of examples)
        num_hidden_layers (class 'int'): number of hidden layers of the network
        num_hidden_units (class 'list'): list containing number of nodes per layer.    
    
    Returns:
        dim_layers (class 'list'): list containing number of nodes per layer, incluiding input and output layers.
    """

    # Define an array to store the network layer structure
    dim_layers = []

    # Define the number of neurons in the input layer
    input_units = X.shape[1]*X.shape[2]*X.shape[3]

    # Add the first layer, in this case, the input layer
    dim_layers.append(input_units)

    # Iterate over the number of hidden layers
    for i in range(0,num_hidden_layers):
        dim_layers.append(num_hidden_units[i])

    # Finally, add the output layer
    output_units = Y.shape[1]
    dim_layers.append(output_units)

    return dim_layers