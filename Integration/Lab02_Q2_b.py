#!/usr/bin/env python
# coding: utf-8

# In[81]:


# Question 2 (b)
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

#Creating the x array and n values for the desired Bessel functions
x = np.linspace(0,20,1000)
n0 = 0
n3 = 3
n5 = 5
#Creating the Bessel functions using our Simpson technique
J0 = Jn(n0,x)
J3 = Jn(n3,x)
J5 = Jn(n5,x)

#Getting the Bessel functions from scipy
from scipy.special import jv
scipy_J0 = jv(0,x) 
scipy_J3 = jv(3,x)
scipy_J5 = jv(5,x)

#plotting our Simpson Rule Bessel functions
plt.plot(x,J0, label = '$J_0(x)$')
plt.plot(x,J3, label = '$J_3(x)$')
plt.plot(x,J5, label = '$J_5(x)$')
plt.legend()
plt.ylabel("Bessel Functions $J_n$(x)")
plt.xlabel("x axis")
plt.title("Bessel Functions created by Simpson's Rule Integration")
plt.xlim(0,20)
plt.show()
#Comparing our Bessel functions against scipy's.
plt.plot(x,J0, label = '$J_0(x)$', linewidth = 4)
plt.plot(x,J3, label = '$J_3(x)$', linewidth = 4)
plt.plot(x,J5, label = '$J_5(x)$', linewidth = 4)
plt.plot(x,scipy_J0, label = '$scipy J_0(x)$', linewidth = 2 , linestyle = ':',color = 'pink')
plt.plot(x,scipy_J3, label = '$scipy J_3(x)$', linewidth = 2 , linestyle = ':', color = 'k')
plt.plot(x,scipy_J5, label = '$scipy J_5(x)$', linewidth = 2 , linestyle = ':', color = 'r')
plt.legend()
plt.ylabel("Bessel Functions $J_n$(x)")
plt.xlabel("x axis")
plt.title("Comparison of our Bessel Functions and Scipy's Bessel Functions")
plt.xlim(0,20)


# In[ ]:





# In[ ]:




