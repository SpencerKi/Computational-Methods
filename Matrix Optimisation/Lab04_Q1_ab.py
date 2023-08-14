# -*- coding: utf-8 -*-
"""
Authors: Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from SolveLinear import GaussElim, PartialPivot

gauss_times = np.empty(0)
pivot_times = np.empty(0)
decom_times = np.empty(0)

gauss_err = np.empty(0)
pivot_err = np.empty(0)
decom_err = np.empty(0)

N = 250

for i in range(5, N):
    A = np.random.rand(i, i)
    v = np.random.rand(i)
    
    start_time = time()
    x = GaussElim(A, v)
    end_time = time()
    
    gauss_times = np.append(gauss_times, end_time - start_time)
    v_sol = np.dot(A, x)
    gauss_err = np.append(gauss_err, np.mean(np.abs(v-v_sol)))
    
    start_time = time()
    x = PartialPivot(A, v)
    end_time = time()
    
    pivot_times = np.append(pivot_times, end_time - start_time)
    v_sol = np.dot(A, x)
    pivot_err = np.append(pivot_err, np.mean(np.abs(v-v_sol)))
    
    start_time = time()
    x = np.linalg.solve(A, v)
    end_time = time()
    
    decom_times = np.append(decom_times, end_time - start_time)
    v_sol = np.dot(A, x)
    decom_err = np.append(decom_err, np.mean(np.abs(v-v_sol)))
    
plt.figure(0)
plt.plot(range(5, N), gauss_times, label = "Gaussian Elimination")
plt.plot(range(5, N), pivot_times, label = "Partial Pivot")
plt.plot(range(5, N), decom_times, label = "LU Decomposition")
plt.yscale("log")
plt.legend()
plt.title("Timing Solutions to Linear Systems")
plt.xlabel("Dimension of Square Matrix (N)")
plt.ylabel("Time to Calculate (log(s))")

plt.figure(1)
plt.plot(range(5, N), gauss_err, label = "Gaussian Elimination")
plt.plot(range(5, N), pivot_err, label = "Partial Pivot")
plt.plot(range(5, N), decom_err, label = "LU Decomposition")
plt.yscale("log")
plt.legend()
plt.title("Error of Solutions to Linear Systems")
plt.xlabel("Dimension of Square Matrix (N)")
plt.ylabel("Relative Error (log scale)")