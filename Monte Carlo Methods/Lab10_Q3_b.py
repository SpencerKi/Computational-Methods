#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 3 b)
# Author: Ilyas Sharif

# Importing required packages
import numpy as np
import matplotlib.pyplot as plt
from random import random

# Define the integrand
def integrand(x):
    return np.exp(-2 * abs(x - 5))
# Defining the integral range
a = 0
b = 10

# Redefining the mean value method from Q3 a)

###################################################
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
###################################################

# Defining the new sampling method for Q3 b)
def sampling_method(N):
    # Initializing output
    Integral = 0
    # Drawing from normal distribution with mean 5, standard devation 1
    # using numpy.random.normal as suggested in lab manual
    x = np.random.normal(5, 1, N)
    # Computing values of our integrand at each x value     
    fx = integrand(x)
    # Using the new weighting function as described by lab manual
    wx = (1 / (2*np.pi)**(1/2)) * np.exp((-(x - 5)**2) / 2)
    # Formula 10.42 from the textbook - calculates integral
    Integral = (sum(fx/wx))/N
    return Integral

# Defining number of sample points and creating arrays for mean value and sampling method
N = 10000
sampling = np.zeros(100)
mean_value = np.zeros(100)


# Repeating the calculation 100 times
for i in range(len(sampling)):
    sampling[i] = sampling_method(N)
    mean_value[i] = mean_value_method(N, a, b)
    
    
# Creating histograms of each integral method ran 100 times with 10000 sample points
# 10 bins - not told range so keep that blank
plt.hist(mean_value, 10)
plt.title(" Histogram of the Mean Value Method ")
plt.ylabel(" Count ")
plt.xlabel(" Computed Value of Integral")
plt.show()
plt.hist(sampling, 10)
plt.title(" Histogram of the Sampling Method ")
plt.ylabel(" Count ")
plt.xlabel(" Computed Value of Integral")
plt.show()


# In[ ]:




