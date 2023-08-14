#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 3
# Author: Ilyas Sharif

from numpy import e
import numpy as np
import matplotlib.pyplot as plt

# Creating our function
def f(x):
    return e**(-1*(x**2))

# Creating our forward difference
def forward(x,h):
    return ( f(x+h) - f(x) )/h

# Creating our centered difference
def centred(x,h):
    return ( f(x+h) - f(x-h) ) / (2*h)

# Defining x, analytical value, and h values
analytical_value = -e**(-1/4)

x = 0.5
h = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

# I will fill this array with values calculated by forward and central difference.
forward_numerical_derivative = np.array([])
centred_numerical_derivative = np.array([])

# This loop simply computes all the forward derivatives for each h value.
for i in range(len(h)):
    a = forward(x,h[i])
    b = centred(x,h[i])
    forward_numerical_derivative = np.append(forward_numerical_derivative,  a)
    centred_numerical_derivative = np.append(centred_numerical_derivative, b)

# Calculates the absolute value of the error
forward_error = np.abs(forward_numerical_derivative - analytical_value)
centred_error = np.abs(centred_numerical_derivative - analytical_value)

print(forward_error)

# Plotting
plt.plot(np.log10(h), np.log10(forward_error), label = "forward difference")
plt.ylabel(" Logarithm of Absolute Error ( Log ($\epsilon$))")
plt.xlabel("Logarithm of Step-size ( Log(h) )")
plt.title(" Error as a Function of Step Size ")
plt.xlim(np.log10(1e-16), 0)
plt.show()
plt.plot(np.log10(h),np.log10(forward_error), label = "forward difference")
plt.plot(np.log10(h),np.log10(centred_error), linestyle = ':', label = "centred difference", color = 'k')
plt.legend()
plt.ylabel(" Logarithm of Absolute Error ( Log ($\epsilon$))")
plt.xlabel("Logarithm of Step-size ( Log(h) )")
plt.title(" Error as a Function of Step Size ")
plt.xlim(np.log10(1e-16), 0)


# In[ ]:




