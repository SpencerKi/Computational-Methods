# -*- coding: utf-8 -*-
"""
Spencer Y. Ki
2023-02-15

Rough work for STA410 CC1
"""
import numpy as np
from numpy.linalg import inv, solve, cholesky, svd, qr
from scipy import stats
import statsmodels.api as sm
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_olivetti_faces

ex = sm.datasets.get_rdataset("mtcars").data.values

def rand_svd(X: np.ndarray) -> np.ndarray:
    """
    Calculates the randomised SVD of X.
    """
    # Centring and scaling X to make Xtilde
    Xtilde = np.empty(X.shape)
    for i in range(len(X.T)):
        col = X.T[i]
        new_col = (col - np.mean(col))/np.std(col)
        Xtilde.T[i] = new_col
    
    # Step 1 (given above)
    p,r = Xtilde.shape[1],5
    np.random.seed(10)
    P = stats.norm().rvs(size=(p,r))
    
    # Step 2
    Z = Xtilde @ P
    
    # Step 3
    Q, R = qr(Z)
    
    # Step 4
    Y = Q.T @ Xtilde
    
    # Step 5
    U, D, V = svd(Y, full_matrices = False)
    
    # Step 6
    return Q, U