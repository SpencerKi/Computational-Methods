# -*- coding: utf-8 -*-
"""
Spencer Y. Ki
2023-02-05

Rough work for STA410 HW4
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot
from numpy.linalg import inv, solve, cholesky, svd, qr
from scipy.linalg import solve_triangular

np.random.seed(10); n,p = 100,10; noise = n/2
X = np.ones((n,p))*np.arange(n)[:,np.newaxis] + stats.norm.rvs(size=(n,p))*noise
X = (X - X.mean(axis=0)[np.newaxis,:])/X.std(axis=0)[np.newaxis,:]
X[:,0] = 1

p1q0 = np.linalg.cond(X.T @ X)

np.random.seed(10); n,p = 100,10; noise = n/5
X = np.ones((n,p))*np.arange(n)[:,np.newaxis] + stats.norm.rvs(size=(n,p))*noise
X = (X - X.mean(axis=0)[np.newaxis,:])/X.std(axis=0)[np.newaxis,:]
X[:,0] = 1

p1q1 = np.linalg.cond(X.T @ X)

# np.random.seed(10); n,p = 100,10; noise,sigma = n/20,1
# X = np.ones((n,p))*np.arange(n)[:,np.newaxis] + stats.norm.rvs(size=(n,p))*noise
# X = (X - X.mean(axis=0)[np.newaxis,:])/X.std(axis=0)[np.newaxis,:]
# X[:,0] = 1; y = X@np.ones((p,1)) + stats.norm.rvs(size=(n,1))*sigma

np.random.seed(10); n,p = 100,10; noise,sigma = .0000001,1
X = np.ones((n,p)) + stats.norm.rvs(size=(n,p))*noise
X[:,0] = 1; y = np.ones((n,1))

p1q2 = inv(X.T @ X) @ X.T @ y
p1q3 = solve(X.T @ X, X.T @ y)
L = cholesky(X.T @ X)
p1q4 = solve_triangular(L.T, solve_triangular(L, X.T @ y, lower=True))
Q,R = qr(X)
p1q5 = solve_triangular(R, Q.T @ y)

U,D,Vt = svd(X, full_matrices=False)
p1q6 = Vt.T @ (inv(np.diag(D)) @ U.T @ y)

p1q7 = "D"
p1q8 = "A"

p1q9 = "C"
p1q10 = "B"

p1q11 = "D"
p1q12 = "B"
p1q13 = "B"
p1q14 = "D" 

betahats_ascols = np.array([[p1q2, p1q3, p1q4, p1q5, p1q6]])
betahats_asrows = np.array([[p1q2], 
                            [p1q3], 
                            [p1q4], 
                            [p1q5],
                            [p1q6]])
np.set_printoptions(linewidth=120)
# Absolute differences in betahat estimates
np.abs(betahats_ascols-betahats_asrows).sum(axis=2).sum(axis=2) # numpy broadcasting
# Rounded beta hat estimates estimates
np.round(np.c_[p1q2, p1q3, p1q4, p1q5, p1q6] ,5)