# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
#from numpy import empty
# The following will be useful for partial pivoting
#from numpy import empty, copy
import numpy as np

def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to temporary variables.
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = np.empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x

def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222) """
    # copy A and v to temporary variables using copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    for m in range(N):
        greatest = 0
        greatest_index = 0
        for j in range(m,N):
            if np.abs(A[j,m]) > greatest:
                greatest = np.abs(A[j,m])
                greatest_index = j
        row_holder = np.copy(A[m])
        digit_holder = np.copy(v[m])
        A[m] = np.copy(A[greatest_index])
        v[m] = np.copy(v[greatest_index])
        A[greatest_index] = np.copy(row_holder)
        v[greatest_index] = np.copy(digit_holder)
        
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = np.empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x