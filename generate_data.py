import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

'''

This script is used for the generation of simulation data. Specifically, 

'''


def sample_uniform_X(n,d):
    X = np.random.uniform(0,1.0,(n,d))
    return X

def sample_boolean_X(n,d):
    X = np.random.randint(0,2.0,(n,d))
    return X

def linear_model(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a linear model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed that 
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i]*beta
        return linear_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train

def sum_of_squares(X,s,beta,sigma):
    
    '''
    This method is used to crete responses from a sum of squares model with hard sparsity
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient vector. If beta not a vector, then assumed that 
    sigma: s.d. of added noise 
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x,s,beta):
        linear_term = 0
        for i in range(s):
            linear_term += x[i]*x[i]*beta
        return linear_term
    y_train = np.array([create_y(X[i, :],s,beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn((len(X)))
    return y_train


# Research Extension
def exponential_model(X, s, beta, sigma):
    '''
    This method is used to create responses from an exponential model with hard sparsity.
    Model: y = exp(beta^T * x) + epsilon
    
    Parameters:
    X: X matrix
    s: sparsity 
    beta: coefficient scalar or array
    sigma: s.d. of added noise 
    
    Returns: 
    numpy array of shape (n)        
    '''
    
    def create_y(x, s, beta):
        linear_term = 0
        # If beta is a scalar or 1D vector:
        is_vector = isinstance(beta, (list, np.ndarray))
        for i in range(s):
            b_i = beta[i] if is_vector else beta
            linear_term += x[i] * b_i
            
        # Apply exponential transformation f(x) = exp(beta^T x)
        return np.exp(linear_term)
    
    y_train = np.array([create_y(X[i, :], s, beta) for i in range(len(X))])
    y_train = y_train + sigma * np.random.randn(len(X))
    return y_train





# def poisson_log_link_mean(X,s,beta):

#     '''
#     This method returns the true conditional mean mu(x) = exp(linear predictor) used by
#     generate_data.poisson_log_link. It is deterministic given X, so calling it with the
#     same X gives the mean that generated y.
#     '''

#     def create_eta(x,s,beta):
#         linear_term = 0
#         for i in range(s):
#             linear_term += x[i]*beta
#         return linear_term
#     eta = np.array([create_eta(X[i, :],s,beta) for i in range(len(X))])
#     return np.exp(eta)

# def log_additive(mu):

#     '''
#     This method draws one sample from the log additive model
#     log mu(x) = beta * sum_{j<s} x_j, y ~ Poisson(mu), x ~ Unif[0,1]^d.
#     '''
    
#     y_train = np.random.poisson(mu)
#     return y_train
