# -*- coding: utf-8 -*-
"""
Spencer Y. Ki
2023-01-29

Rough work for STA410 HW3
"""
import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from scipy import linalg
import warnings

def is_linearly_independent_columns(X):
    return min(X.shape) == np.linalg.matrix_rank(X)


def gram_schmidt(X, method="classic"):
    """
    X       : p linearly independent column vectors
              (np.array) [X[:,0], X[:,1], ... , X[:,p-1]]), X.shape=(n,p)
              or raises a "Linearly Dependent Columns" ValueError 

    method  : (str) <"modified"|"classic">

    returns : p orthonormalized column vectors, 
              (np.array) [Xtilde[:,0], Xtilde[:,1], ... , Xtilde[:,p-1]], Xtilde.shape=(n,p)
    """
    
    if not is_linearly_independent_columns(X):
        raise ValueError('Linearly Dependent Columns')

    X = np.array(X, dtype=float)
        
    if method=="modified":
        X.T[0] = X.T[0] * (X.T[0] @ X.T[0]) ** (-0.5)
        for k in range(1, len(X.T)):
            for j in range(k, len(X.T)):
                X.T[j] = X.T[j] - X.T[k-1] * (X.T[j] @ X.T[k-1])
            X.T[k] = X.T[k] * (X.T[k] @ X.T[k]) ** (-0.5)
        return X
    
    elif method=="classic":
        X_bar = X
        for k in range(len(X.T)):
            sig = X_bar.T[0] * (X_bar.T[0] @ X.T[k])
            for j in range(1, k):
                sig += X_bar.T[j] * (X_bar.T[j] @ X.T[k])
            X_bar.T[k] = X.T[k] - sig
            X.T[k] = X.T[k] * (X.T[k] @ X.T[k]) ** (-0.5)
        return X_bar
    
    else:
        raise ValueError("Unknown Method")
        
# def cholesky_at_kth_step(A, k):
#     """
#     A       : (np.array) a square matrix with n columns
#     k       : (int) the step of the factorization process of the Cholesky 
#             "inner product" algorithm one wants returned

#     returns : (np.array) the matrix Qk with the same dimensions as A that is 
#     the result of transforming A by k steps of the Cholesky "inner product" algorithm
#     """
#     n = min(A.shape)
#     if np.linalg.matrix_rank(A) < n:
#         raise ValueError("A is not full rank")
#     if k < 0 or k > n:
#         raise ValueError("k must be a nonnegative integer from 0 to n")
    
#     n = np.linalg.matrix_rank(A)
#     Qk = 0*A
#     #<complete "inner product" algorithm which stops after the kth step as described>
#     step = 0
#     while step <= k:
#         if step == 0:
#             Qk[0,0] = np.sqrt(A[0, 0])
#             step += 1
#         elif step in range(1, n):
#             Qk[0, step] = A[0, step]
#             step += 1
#         else:
#             for i in range(1, n):
#                 sig = 0
#                 for r in range(i):
#                     sig += Qk[r,i]**2
#                 Qk[i,i] = np.sqrt(A[i,i] - sig)
#                 step += 1
#                 for j in range (i+1, n):
#                     sig = 0
#                     for r in range(i):
#                         sig += Qk[r,i] * Qk[r,j]
#                     Qk[i,j] = (A[i,j] - sig)/Qk[i,i]
#                     step += 1
#     return Qk

def cholesky_at_kth_step(A, k):
    """
    A       : (np.array) a square matrix with n columns
    k       : (int) the step of the factorization process of the Cholesky 
            "inner product" algorithm one wants returned

    returns : (np.array) the matrix Qk with the same dimensions as A that is 
    the result of transforming A by k steps of the Cholesky "inner product" algorithm
    """
    n = A.shape[0]
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("A is not full rank")
    if k < 0 or k > n:
        raise ValueError("k must be a nonnegative integer from 0 to n")
    #<raise appropriate `ValueError`s regarding inappropriate `A` and `k`>
    #<regarding `k` only allow nonnegative integers from 0 to `K` which produce unique results>
    
    Qk = 0*A
    #<complete "inner product" algorithm which stops after the kth step as described>
    Qk[0,0] = np.sqrt(A[0,0])
    for j in range(1, k):
        Qk[0,j] = A[0,j]/Qk[0,0]

    for i in range(1, k):
        urisq = 0
        for r1 in range(i):
            urisq += Qk[r1,i]**2
        Qk[i,i] = np.sqrt(A[i,i] - urisq)
        for j in range(i+1, k):
            uriruj = 0
            for r2 in range(i):
                uriruj += Qk[r2,i]*Qk[r2,j]
            Qk[i,j] = (A[i,j] - uriruj)/Qk[i,i]
    return Qk

thrown_errors = 0
try:
    cholesky_at_kth_step(np.ones((2,2)), 2)
except:
    thrown_errors +=1
try:
    cholesky_at_kth_step(np.eye(2), 4)
except:
    thrown_errors +=1
try:
    cholesky_at_kth_step(np.eye(2), 3)
except:
    thrown_errors +=1        
try:
    cholesky_at_kth_step(np.eye(2), 0.5)
except:
    thrown_errors +=1
try:
    cholesky_at_kth_step(np.ones((2,1)), 1)
except:
    thrown_errors +=1
 
A = np.diag(np.ones(6))+0.1
k1_good = np.isclose(cholesky_at_kth_step(A, k=1), cholesky_at_kth_step(A, k=1)).all()
k7_good = np.isclose(cholesky_at_kth_step(A, k=7), cholesky_at_kth_step(A, k=7)).all()
k17_good = np.isclose(cholesky_at_kth_step(A, k=17), cholesky_at_kth_step(A, k=17)).all()
k21_good = np.isclose(cholesky_at_kth_step(A, k=20), cholesky_at_kth_step(A, k=20)).all()
