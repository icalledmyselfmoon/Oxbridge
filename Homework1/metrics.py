import numpy as np

#fp len([i for i, j in zip(y_true, y_pred) if (i != j & j == 1)])
#fn len([i for i, j in zip(y_true, y_pred) if (i != j & j == 0)])
#tp len([i for i, j in zip(y_true, y_pred) if i == j ])



def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    fp = len([i for i, j in zip(y_true, y_pred) if ((i != j) & (j == 1))])
    fn = len([i for i, j in zip(y_true, y_pred) if ((i != j) & (j == 0))])
    tp = len([i for i, j in zip(y_true, y_pred) if i == j ])
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2/((1/recall) + (1/precision))
    accuracy = tp/len(y_pred)
    
    return precision, recall, f1, accuracy 


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    fp = len([i for i, j in zip(y_true, y_pred) if ((i != j) & (j == 1))])
    fn = len([i for i, j in zip(y_true, y_pred) if ((i != j) & (j == 0))])
    tp = len([i for i, j in zip(y_true, y_pred) if i == j ])
    
    accuracy = tp/len(y_pred)
    return accuracy 


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    squared_error = (y_pred - y_true)**2
    
    #Finding the mean squared error:
    error_var = squared_error.sum()
    sample_var = ((y_true - y_true.mean())**2).sum()
    r = (1 - (error_var / sample_var))
    return r
    


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.mean(np.power((y_pred - y_true),2))
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.mean(np.abs((y_pred - y_true)))
    return mae
    
