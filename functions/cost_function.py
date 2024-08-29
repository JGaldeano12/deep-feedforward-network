import numpy as np

def cost_function(predictions, Y):
    """_summary_
    Function that calculates the error between the model's predictions and the correct labels.

    Args:
        predictions (class 'numpy.ndarray'): network predictions.
        Y (class 'numpy.ndarray'): true labels.

    Returns:
        cost (class 'numpy.float64'): error obtained from network predictions compared to true labels.
    """

    # Calculate the loss function
    cost = - np.mean( Y * np.log(predictions) + (1-Y) * np.log(1-predictions))

    cost = np.squeeze(cost) 

    return cost