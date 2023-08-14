#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2 (a) iii.
# Author: Ilyas Sharif
# We will build off of trapezoidal.py for this problem
from numpy import pi 
#import the "time" function from the time module
from time import time

#Defining our function
def f(x):
    return 4 / (1 + x**2)

# As recommened, we will use the N = 2^n approach and increase n until we find O(10e-9)
#I created a n and N for both the simpson and trapezoidal method making sure to select the values
#that create the O(10e-9)
n_simp = 4
n_trap = 12
N_simp = 2**n_simp
N_trap = 2**n_trap
#N_simp = N_trap
a = 0.0 #starting point
b = 1.0 #end point
h_simp = (b-a)/N_simp #width of slice for simpson to reach O(10e-9)
h_trap = (b-a)/N_trap #width of slice for trap to reach O(10e-9)
time_Trapezoidal = 0.
time_Simpson = 0.

# Runs through each integration 100 times
for i in range(100):
    start_Trapezoidal = time()
    # Trapezoidal Rule - find inital points, sum over the middle points.
    Trapezoidal = 0.5*f(a) + 0.5*f(b) #inital points
    for k in range(1,N_trap): #the sum term in trapezoidal rule
        Trapezoidal += f(a+k*h_trap)
    #must multiply by h to get value of integral
    Trapezoidal = h_trap*Trapezoidal
    end_Trapezoidal = time()
    time_Trapezoidal += end_Trapezoidal - start_Trapezoidal


    start_Simpson = time()
    # Simpson Rules - find inital points, do 2 sums, one for even, one for odd
    Simpson = f(a) + f(b) #inital points
    for k in range(1,N_simp,2): #the odd sum in simpson's rule
        Simpson += f(a+k*h_simp)*4
    for k in range(2,N_simp,2): #the even sum in simpson's rule
        Simpson += f(a+k*h_simp)*2
    #must multiply by h/3 to get proper value of integral
    Simpson = (h_simp/3)*Simpson
    end_Simpson = time()
    time_Simpson += end_Simpson - start_Simpson


# Print Statements
#Calulated values
print('Trapezoidal rule gives us an approximate value of '+ str(Trapezoidal)) 
print( 'Simpson\'s rule gives us an approximate value of '+ str(Simpson))
#Difference with the exact value
print('Trapezoidal rule is off by '+ str(pi -Trapezoidal))
print( 'Simpson\'s rule is off by '+ str(pi - Simpson))
# Time it took to run the integration
print('Trapezoidal rule average integration time is '+ str(time_Trapezoidal/100))
print( 'Simpson\'s rule average integration time is '+ str(time_Simpson/100))     


# In[ ]:




