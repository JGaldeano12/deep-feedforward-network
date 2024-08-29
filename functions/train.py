import numpy as np
from init_parameters import init_parameters
from forward_activation import forward_activation
from cost_function import cost_function
from backward_propagation import backward_propagation
from update_parameters import update_parameters

def train_model(X, Y, val_x, val_y, model, lrate, num_iterations):
    """_summary_
    Function that optimize the neural network.

    Args:
        X (class 'numpy.ndarray'): training dataset.
        Y (class 'numpy.ndarray'): training labels.
        val_x (class 'numpy.ndarray'): validation dataset.
        val_y (class 'numpy.ndarray'): validation labels.
        model (class 'list'): list containing number of nodes per layer, including input and output layers.
        lrate (class 'numpy.float64'): learning rate value.
        num_iterations (class 'int'): number of training iterations through dataset.

    Returns:
        parameters (class 'dict'): dictionary consisting of the set of updated parameters from the network after training.
        loss_train (class 'list'): list of training error progression throughout optimization.
        loss_val (class 'list'): list of validation error progression throughout optimization.
        error (class 'numpy.float64'): error obtained on the last iteration, which means the error of the network with training dataset.
    """

    gradients = {}
    
    # To store the error after each iteration
    loss_train = []
    loss_val = []

    # Adjust the shape of the labels
    Y = Y.T
    val_y = val_y.T

    # 1.- Initialize the parameters
    parameters = init_parameters(model)

    # 2.- Training iteration
    for i in range(0, num_iterations):
        
        # First, call 'forward_activation'
        AL, cache = forward_activation(X, parameters)
        
        # For validation:
        AL_val, cache_val = forward_activation(val_x, parameters)

        # Adapt the predictions to avoid errors when calculating logarithms
        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
        
        # Compute the current error
        error = cost_function(AL, Y)
        val_error = cost_function(AL_val, val_y)

        # Initialize 'backward propagation'
        gradients = backward_propagation(AL, Y, cache)
        
        # Update the parameters to accelerate training
        parameters = update_parameters(parameters, gradients, lrate)

        # Display the error in the console
        print("Epoch ", i, " --- Training Error: ", error, " --- Val. Error: ", val_error)
        
        # Save the error of the iteration
        loss_train.append(error)
        loss_val.append(val_error)
    
    # Return the network parameters after training
    return parameters, loss_train, loss_val, error