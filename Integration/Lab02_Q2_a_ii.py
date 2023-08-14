#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Question 2 (a) ii.
# Author: Ilyas Sharif
# We will build off of trapezoidal.py for this problem
from numpy import pi 

#Defining our function
def f(x):
    return 4 / (1 + x**2)

N = 4
a = 0.0 #starting point
b = 1.0 #end point
h = (b-a)/N #width of slice

# Trapezoidal Rule - find inital points, sum over the middle points.
Trapezoidal = 0.5*f(a) + 0.5*f(b) #inital points
for k in range(1,N): #the sum term in trapezoidal rule
    Trapezoidal += f(a+k*h)
#must multiply by h to get value of integral
Trapezoidal = h*Trapezoidal


# Simpson Rules - find inital points, do 2 sums, one for even, one for odd
Simpson = f(a) + f(b) #inital points
for k in range(1,N,2): #the odd sum in simpson's rule
    Simpson += f(a+k*h)*4
for k in range(2,N,2): #the even sum in simpson's rule
    Simpson += f(a+k*h)*2
#must multiply by h/3 to get proper value of integral
Simpson = (h/3)*Simpson



# Print Statements
#Calulated values
print('Trapezoidal rule gives us an approximate value of '+ str(Trapezoidal)) 
print( 'Simpson\'s rule gives us an approximate value of '+ str(Simpson))
#Difference with the exact value
print('Trapezoidal rule is off by '+ str(pi -Trapezoidal))
print( 'Simpson\'s rule is off by '+ str(pi - Simpson))
#Percent error with exact value
print('Trapezoidal rule percent error is '+ str  (abs((Trapezoidal - pi)/pi)*100) + '')
print( 'Simpson\'s rule percent error is '+ str  (abs((Simpson - pi)/pi)*100) + '%')     

