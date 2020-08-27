import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is 
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        estimates(np.ndarray): the estimated values (should be the same shape as targets)
        targets(np.ndarray): the ground truth values

    Returns:
        MSE(int): mean squared error calculated by above equation 
    """
    diffs = []
    if estimates.ndim == 1:
        for n in range(estimates.size):
            diff = estimates[n] - targets[n]
            diffs.append(diff)
    else:
        for i in range(estimates.shape[0]):
            for j in range(estimates.shape[1]):
                diff = estimates[i][j] - targets[i][j]
                diffs.append(diff)
    diffs = np.array(diffs)
    diffs = np.square(diffs)
    error = np.mean(diffs)
    return error