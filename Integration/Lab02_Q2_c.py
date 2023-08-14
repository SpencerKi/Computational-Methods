#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Question 2 (c)
# Author: Ilyas Sharif
# We will build off of trapezoidal.py for this problem
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos, sin

#Defining the integrand of the bessel function - I will divide by pi later
def f(n,x,phi):
    return cos(n*phi - x*sin(phi))

#Defining the constants required for the Simpson Rule Integration
N = 1000
a = 0.0 #starting point
b = pi #end point
h = (b-a)/N #width of slice
n = 2
z = 11.620

#Defining our actual bessel function (applying the Simpson Rule integration to get Jn(x))
# We are basically creating a new function that creates a desired Bessel function, when an array (x)
# and an n value are inputted. 
def Jn(n,x):
    
    # Simpson Rules - find inital points, do 2 sums, one for even, one for odd
    Simpson = f(n,x,a) + f(n,x,b) #inital points
    for k in range(1,N,2): #the odd sum in simpson's rule
        Simpson += f(n,x,a+k*h)*4
    for k in range(2,N,2): #the even sum in simpson's rule
        Simpson += f(n,x,a+k*h)*2
    #must multiply by h/(3 pi) to get proper value of integral
    Simpson = (h/(3*pi))*Simpson
    return Simpson

def umn(n,z,r,theta):
    R0 = (50)**(1/2) # note in order to obey 0 =< r/R =< 1, then R0 must be the max value which is sqrt(5^2 + 5^2)
    umn = Jn(n,(z/R0)*r)*cos(n*theta)
    # i ignored the cos(c*z*t/a) part since t = 0 will set that term equal to 1.
    return umn

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = umn(n,z,R, np.arcsin(Y/R))
Z[R>50**(1/2)] = 0.0

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, pad = 0.2)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis - Umn(r,$\theta$)')
ax.view_init(-10, 0)
plt.show()


# In[ ]:




