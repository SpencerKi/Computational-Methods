# -*- coding: utf-8 -*-
"""
Authors: Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt

# Non-linear function
def unsolvable(c, x):
    return 1 - np.exp(-c*x)

# Relaxation method
def relax(c, guess):
    est = unsolvable(c, guess)
    counter = 0
    while np.abs(unsolvable(c, est) - est) > 1e-6:
        est = unsolvable(c, est)
        counter += 1
    return est, counter

# Overrelaxation method
def overrelax(c, guess, w):
    est = unsolvable(c, guess)
    counter = 0
    while np.abs(unsolvable(c, est) - est) > 1e-6:
        est = (1 + w)*unsolvable(c, est) - w*est
        counter += 1
    return est, counter

# Plot
if __name__ == "__main__":
    x_values = np.empty(0)
    for i in np.arange(0, 3.01, 0.01):
        x_values = np.append(x_values, relax(i, 1)[0])
    
    plt.figure(0)
    plt.plot(np.arange(0, 3.01, 0.01), x_values)
    plt.title("x as a function of c")
    plt.xlabel("c")
    plt.ylabel("x")