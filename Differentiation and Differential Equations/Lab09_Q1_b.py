#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 b)
# Author: Ilyas Sharif

# Importing required packages
from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt
import numpy as np

# Copying Newman's functions so no need for external files

######################################################################
# 1D DST Type-I

def dst(y):
    N = len(y)
    y2 = empty(2*N,float)
    y2[0] = y2[N] = 0.0
    y2[1:N] = y[1:]
    y2[:N:-1] = -y[1:]
    a = -imag(rfft(y2))[:N]
    a[0] = 0.0

    return a

######################################################################
# 1D inverse DST Type-I

def idst(a):
    N = len(a)
    c = empty(N+1,complex)
    c[0] = c[N] = 0.0
    c[1:N] = -1j*a[1:]
    y = irfft(c)[:N]
    y[0] = 0.0

    return y

######################################################################

# Defining constants
v = 100
L = 1
d = .1
C = 1
sigma = 0.3
# time will be varied from 2,4,6,12,100 ms (converts to seconds)
t = [2/1000, 4/1000, 6/1000, 12/1000, 100/1000]


# Defining problem parameters
nx = 20000
x = np.arange(0, L, L/nx)


# Defining functions

# initial condition that phi(x) = 0 everywhere...
def phi0(x):
    return 0

# ...but the velocity psi(x) is nonzero, with profile psi(x)
def psi0(x):
    return (C * ( x * (L-x)) / L**2)* np.exp(-((x-d)**2)/(2*sigma**2))

# Our solution for phi(x,t)
def phi(phik,psik, t):
    phixt = np.empty(len(psik))
    for k in range(len(phik)):
        wk = (pi*v/L)*(k)
        phixt[k] = (phik[k] * np.cos(wk*t)) + (psik[k]/wk * np.sin(wk*t))
    return (idst(phixt))

# Creating coefficents
phix = np.empty(len(x))
psix = np.empty(len(x))

for i in range(len(x)):
    phix[i] = phi0(x[i])
    psix[i] = psi0(x[i])

# Fourier transforming the coefficents
phik = dst(phix)
psik = dst(psix)

# Plotting all solutions
for t in t:
    plt.plot(x, phi(phik, psik, t), label = "t = " + str(t))

# Creating legend, axes, etc.
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Solution to One Dimensional Wave Equation Using Spectral Method")

