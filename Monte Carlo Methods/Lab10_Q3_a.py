#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Question 3 a)
# Author: Ilyas Sharif

# Importing required packages
import numpy as np
import matplotlib.pyplot as plt
from random import random

# Define the integrand
def integrand(x):
    return (x**(-1/2)) / (1 + np.exp(x))
# Defining the integral range
a = 0
b = 1


# Define the sampling method
def sampling_method(N, a, b):
    # Initializing output and x array
    Integral = 0
    x = np.zeros(N)
    # Getting our "random" x values 
    for i in range(N):
        x[i] = (a + (random() * (b - a)))**2  
    # Computing values of our integrand at each x value     
    fx = integrand(x)
    # Using the weighting function as described by lab manual
    wx = x**(-1/2)
    # Formula 10.42 from the textbook - calculates integral
    Integral = (sum(fx/wx))*2/N
    return Integral

# Define the mean value method
def mean_value_method(N,a,b):
    # Initializing output and x array
    Integral = 0
    x = np.zeros(N)
    # Getting our "random" x values
    for i in range(N):
        x[i] = (random() * (b - a)) + a
    # Computing values of our integrand at each x value    
    fx = integrand(x)
    # Equation 10.30 from the textbook - calculates integral
    Integral = (sum(fx))*(b-a)/N
    return Integral
 
# Defining number of sample points and creating arrays for mean value and sampling method
N = 10000
sampling = np.zeros(100)
mean_value = np.zeros(100)

# Repeating the calculation 100 times
for i in range(len(sampling)):
    sampling[i] = sampling_method(N, a, b)
    mean_value[i] = mean_value_method(N, a, b)

# Creating histograms of each integral method ran 100 times with 10000 sample points
plt.hist(mean_value, 10, range=[0.8, 0.88])
plt.title(" Histogram of the Mean Value Method ")
plt.ylabel(" Count ")
plt.xlabel(" Computed Value of Integral")
plt.xlim(0.8, 0.88)
plt.show()
plt.hist(sampling, 10, range=[0.8, 0.88])
plt.title(" Histogram of the Sampling Method ")
plt.ylabel(" Count ")
plt.xlabel(" Computed Value of Integral")
plt.xlim(0.8, 0.88)
plt.show()


# In[ ]:




