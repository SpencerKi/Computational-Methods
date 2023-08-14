#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 c.
# Author: Ilyas Sharif


# *** Attention: Please be patient with running this code
# *** I use 2 nested for loops to compute the multi-dimensional array for intensity
# *** It takes around 40-50 seconds to generate each plot
# *** I only create 1 plot per run to save computing power - simply change Lambda to view all plots.
# *** This is slightly different from question 1 b ii) because not having the gauassian quadrature as a function
# *** greatly impacted the performance of my results, so i just moved that code to be a function.

import numpy as np
from numpy import pi
from numpy import cos
from numpy import sin
from gaussxw import gaussxw
import matplotlib.pyplot as plt
import scipy.special as sc

#Defining our fresnal functions
def C(t):
    return cos( (1/2) * pi * t**2 )
def S(t):
    return sin( (1/2) * pi * t**2 )

#Defining the Gaussian Quadrature method
def Gauss_Quad(N,a,b,f):
    #calculates weights and points - maps them to appropriate interval [a,b]
    x,w = gaussxw(N)
    xp = 0.5*(b)*x + 0.5*(b)
    wp = 0.5*(b)*w
    # Perform the integration
    s = 0
    for k in range(N):
        s += wp[k]*f(xp[k])   
    return s



N = 50 # number of slices
a = 0.0 #starting point
Lambda = 2
# x and z array
X = np.linspace(-3,10,100)
Z = np.linspace(1,5,100)
X, Z = np.meshgrid(X,Z)
# this is point b
u = X*np.sqrt( 2 / (Lambda * Z))
# this array will hold the intensity points
I = np.empty_like(u)


     
for i in range(len(X)):
    for j in range(len(Z)):
        Cu = Gauss_Quad(N,a,u[i][j],C)
        Su = Gauss_Quad(N,a,u[i][j],S)
        I[i][j] = ((1/8)*( (2*Cu + 1)**2 + (2*Su + 1)**2 ))

plt.pcolormesh(Z, X, I)
plt.xlabel(" Z axis - Horizontal Distance from Object")
plt.ylabel(" X axis - Vertical Distance from Object")
plt.title(" 2D Intensity Pattern $\lambda$ = " + str(Lambda))
plt.colorbar()
plt.show()


# In[ ]:




