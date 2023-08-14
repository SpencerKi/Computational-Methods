#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Question 2 (a) iv.
# Author: Ilyas Sharif
# We will build off of trapezoidal.py for this problem
from numpy import pi 

#Defining our function
def f(x):
    return 4 / (1 + x**2)

N1 = 16
N2 = 2*N1
a = 0.0 #starting point
b = 1.0 #end point
h1 = (b-a)/N1 #width of slice
h2 = (b-a)/N2 

# Trapezoidal Rule - find inital points, sum over the middle points.
# Evaluating integrals to get I1 and I2

Trapezoidal1 = 0.5*f(a) + 0.5*f(b) #inital points
for k in range(1,N1): #the sum term in trapezoidal rule
    Trapezoidal1 += f(a+k*h1)
    
Trapezoidal2 = 0.5*f(a) + 0.5*f(b) #inital points
for k in range(1,N2): #the sum term in trapezoidal rule
    Trapezoidal2 += f(a+k*h2)
    
#must multiply by h to get value of integral
Trapezoidal1 = h1*Trapezoidal1
Trapezoidal2 = h2*Trapezoidal2

#The error estimation for the second integration is given by epsilon = (I2 - I1)/3
# As given by equation 5.28 in the text
error = (Trapezoidal2 - Trapezoidal1)/3


# Print Statements
#Calulated values
print('Trapezoidal rule for N = 16 gives us an approximate value of '+ str(Trapezoidal1)) 
print('Trapezoidal rule for N = 32 gives us an approximate value of '+ str(Trapezoidal2)) 
print('The error estimation for N = 32 is ' + str(error))

  


# In[ ]:




