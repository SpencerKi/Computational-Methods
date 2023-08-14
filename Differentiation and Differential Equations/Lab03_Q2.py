# -*- coding: utf-8 -*-
"""
Spencer Ki
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from gaussxw import gaussxwab

# Constants given by problem
m = 1
k = 12

# Velocity function given in physics background
def velocity(x, x_0):
    return c*np.sqrt((k*(x_0**2-x**2)*(2*m*c**2+k*(x_0**2-x**2)/2))/\
                     (2*(m*c**2+k*(x_0**2-x**2)/2)**2))

# Integrand for period
def integrand(x, x_0):
    return 4*velocity(x, x_0)**-1

# Integrating the above integrand via Gaussian Quadrature
# Modified from the example given in the textbook
def period(x_0, N):
    x,w = gaussxwab(N, 0, x_0)
    s = 0.0
    for i in range(N):
        s += w[i]*integrand(x[i], x_0)
    return s

# Estimating error via the method given in the Computational background
def est_error(x_0, N):
    return period(x_0, 2*N) - period(x_0, N)

# Calculating periods by various values of N
T_boring = 2*np.pi*np.sqrt(m/k)
T_classical = period(0.01, 1000)
T_8 = period(0.01, 8)
T_16 = period(0.01, 16)

# Fractional errors
t_8_err = est_error(0.01, 8) / period(0.01, 8)
t_16_err = est_error(0.01, 16) / period(0.01, 16)
t_200_err = est_error(0.01, 200) / period(0.01, 200)

# Sampling points required for Gaussian Quadrature
points_8 = gaussxwab(8, 0, 0.01)
points_16 = gaussxwab(16, 0, 0.01)

# Unweighted sampling point plot
plt.figure(0)
plt.scatter(points_8[0], integrand(points_8[0], 0.01), label = "N = 8")
plt.scatter(points_16[0], integrand(points_16[0], 0.01), label = "N = 16")
plt.xlabel("Sampling Points")
plt.ylabel("Integrands 4/g_k")
plt.title("Period Integrand Points by Gaussian Quadrature")
plt.legend()

# Weighted sampling point plot
plt.figure(1)
plt.scatter(points_8[0], integrand(points_8[0], 0.01)*points_8[1], label = "N = 8")
plt.scatter(points_16[0], integrand(points_16[0], 0.01)*points_16[1], label = "N = 16")
plt.xlabel("Sampling Points")
plt.ylabel("Weighted Integrands 4w_k/g_k")
plt.title("Weighted Period Integrand Points by Gaussian Quadrature")
plt.legend()

# Calculated initial displacement for v = c at x = 0
x_c = c/np.sqrt(k/m)

# Various periods from 1m < x_0 < 10*x_c
h = np.zeros(1)
for i in np.linspace(1, 10*x_c, 50):
    h = np.append(h, period(i, 200))
h = np.delete(h, 0)

# Period vs x_0
plt.figure(2)
plt.scatter(np.linspace(1, 10*x_c, 50), h)
plt.xlabel("Initial Displacement (m)")
plt.ylabel("Period (s)")
plt.title("Period as a Function of Initial Displacement")