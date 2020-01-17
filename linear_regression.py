"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
import math

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
   
    #X_transpose = X.T
    #print(X.shape)
    #print(w.shape)
    
    err = None
    predict = X.dot(w)
    #num_samples , D = X.shape
    err = float((np.mean(np.absolute(predict - y))))
  
    #predict = X.dot(w)
    #err = (np.absolute(predict - y)**2).mean()
    
    return err
    #err = None
   

###### Q1.2 ######
def linear_regression_noreg(X, y):
    
    
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    
    X_transpose = X.transpose()
    
    X_transposeX= X_transpose.dot(X)
    
    Inverse_X_transposeX = np.linalg.inv(X_transposeX)
    dot_product = X_transpose.dot(y)
    
    w = Inverse_X_transposeX.dot(dot_product)
    #w = None
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
                     

    #matrix = X.T.dot(X)
    X_transpose = X.transpose()
    
    X_transposeX= X_transpose.dot(X)
    n,m = X_transposeX.shape
    i_dimension = n
    
    I = 0.1*np.identity(i_dimension)
    eigenvalues, eigenvectors = np.linalg.eig(X_transposeX)
    check_value = np.abs(eigenvalues).min()
    
    while check_value < 0.00001:
        X_transposeX = X_transposeX+ I
        eigenvalues, eigenvectors = np.linalg.eig(X_transposeX)
        check_value = np.abs(eigenvalues).min()
    
    X_transposeY = X_transpose.dot(y)
    inverse_matrix =np.linalg.inv(X_transposeX)
    w = inverse_matrix.dot(X_transposeY)
    
    
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    #####################################################	

    w = None
    X_transpose = X.transpose()
    
    X_transposeX= X_transpose.dot(X)
    n,m = X_transposeX.shape
    i_dimension = n
    #i_dimension = X_transposeX.shape[0]
    I = lambd*np.identity(i_dimension)
    inverse_matrix = np.linalg.inv(X_transposeX+I)
    X_transposeY = X_transpose.dot(y)
    w = inverse_matrix.dot(X_transposeY)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    
    mae_base = float('inf')
    
    i = -19
    
    while i < 20:
        
        
        lambd = math.pow(10,i)
        
        weights_vector = regularized_linear_regression(Xtrain, ytrain, lambd)
        
        error = mean_absolute_error(weights_vector, Xval, yval)
        if error < mae_base:
            bestlambda = lambd
            mae_base = error
        i+=1
        

    
    
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    
    #Change here
    X_array = [] 
    X_array.append(X)
    i = 1
    
    while i < power:
        temp = X_array[i-1]
        X_array.append(temp*X)
        i +=1
    X = np.hstack((X_array))
   
    
    
    return X


