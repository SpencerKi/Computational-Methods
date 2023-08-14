#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 (b) ii.
# Author: Ilyas Sharif

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




a = 0.0 #starting point
# defining our x array. In order to keep the x variables from gaussxw seperate
# we define this x as xx

# initializing the delta arrays

max_delta_array = []

    # Running a for loop over N from 3 to 50 so i don't need to copy paste 47 times.
N_range = range(3,51,1)

for i in N_range:
    N = i
    I_G = []
    xx = np.linspace(-5, 5, 50, endpoint = True)
    # interating over all values for a in the range from -5m to 5m
    for i in range(len(xx)):
        #for each x value, we need to calculate a value for  u. This is the upperbound of the integral.
        u = xx[i] * np.sqrt( 2 / 3) # lambda = 1 and z = 3.
    
        # Gaussian Quadrature - Copied style from gaussint.py
        # Calculate the sample points and weights, then map them
        # to the required integration domain
        x,w = gaussxw(N)
        xp = 0.5*(u-a)*x + 0.5*(u+a)
        wp = 0.5*(u-a)*w
        
        # Perform the integration
        # These are termed Ci / Si where the i indicates the inital values required for Gaussian Quadrature
        Ci = 0.0
        Si = 0.0
        
        # Gaussian Quadrature code
        for k in range(N):
            Ci += wp[k]*C(xp[k])
            Si += wp[k]*S(xp[k])
            
        # Now that we have values for C(u) and S(u), we can
        # calculate our value for I and append to the I array
        I_G.append( (1/8)*( (2*Ci + 1)**2 + (2*Si + 1)**2 )  )

    
    # Now, repeating the entire process with Scipy's functions.
    u_scipy = xx * np.sqrt( 2 / 3)
    S_scipy, C_scipy = sc.fresnel(u_scipy)
    I_SP = (1/8)*( (2*C_scipy + 1)**2 + (2*S_scipy + 1)**2 )
    
    #Getting values for delta
    delta = (abs(I_SP - I_G))  /I_SP
    delta = delta[24:] #This makes it ignore any values for x < 0
    max_delta = max(delta)
    max_delta_array.append(max_delta)

plt.plot(N_range, max_delta_array)
plt.xlim(3,50)
plt.ylabel(" max $\delta$(N)")
plt.xlabel("N - Number of Slices")
plt.title("Maximum Value of the Relative Difference as a Function of N")

