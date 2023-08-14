#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 3
# Author: Ilyas Sharif

from numpy import empty,array,arange
import numpy as np
import matplotlib.pyplot as plt

# Defining our initial constants.

constant_a = 1 # Constants used in ODEs
constant_b = 3 # Constants used in ODEs

a = 0.0 # starting time
b = 20.0 # ending time
N = 1          # Number of "big steps" (start with 1 step)
H = (b-a)/N      # Size of "big steps"
delta = 1e-10     # Required position accuracy per unit time


# initial x and y points are 0, so define the r array
# with the inital x and y points as 0
r = array([0.0,0.0],float) 
time = [0]
x = [0]
y = [0]



# Defining our ODEs
def f(r):
    x = r[0]
    y = r[1]
    fx = 1 - (constant_b + 1)*x + constant_a*(x**2)*y
    fy = constant_b*x - constant_a*(x**2)*y
    return array([fx,fy],float)

def step(r, t, H):
    # Going through one modified midpoint step to start things off
    # same as bulirsch.py
    n = 1
    r1 = r + 0.5*H*f(r)
    r2 = r + H*f(r1)
    R1 = np.empty([1,2],float)
    R1[0] = 0.5 * (r1+r2+0.5*H*f(r2))
    
    # going through each segment for high error and a maximum
    # of 8 modified midpoint steos in our interval before we splice again
    # in modified bulirsch.py, Newman doesn't need the extra condition
    while 2*H*delta > H*delta and n <= 8:
        # Calculating the modified midpoint step for the further rows
        n += 1
        h = H/n
        r1 = r + 0.5*h*f(r)
        r2 = r + h*f(r1)
        for i in range(n - 1):
            r1 += h*f(r2)
            r2 += h*f(r1)
        # We can now get the values for the modified midpoint steps to use in the
        # table as seen on pg.380 of the textbook
        R2 = R1
        R1 = np.empty([n,2], float)
        R1[0] = 0.5* (r1 + r2 + 0.5 *h*f(r2))
        
        # We can now explictlity go and calculate the values for all the rows
        # as well as the error
        for m in range(1,n):
            epsilon = (R1[m-1] - R2[m-1])/((n/(n-1))**(2*m)-1)
            R1[m] = R1[m-1] + epsilon
        error = (epsilon[0]**2 + epsilon[1]**2)**(1/2)
        
    # the maximum error we allow
    target_error = h*delta
    
    if error <= target_error:
        time.append(t + H)
        x.append(R1[n - 1][0])
        y.append(R1[n - 1][1])
        return R1[n-1]
    else:
        # Recurrsion in the Bulirsch–Stoer method as defined in the lab handout
        r1 = step(r,t,H/2)
        r2 = step(r1, t+H/2, H/2)
        return r2
    
# Getting the answer    
answer = step(r, 0, 20)

# Plotting both
plt.plot(time, x, color = 'k' , label = 'x')
plt.plot(time, x, '.', color = 'k')
plt.plot(time, y, color = 'r', label = 'y')
plt.plot(time, y, '.', color = 'r')
plt.legend()
plt.xlim(0,20)
plt.xlabel("Time")
plt.ylabel(" Concentration of Chemicals")
plt.title("Brusselator - Chemical Oscillator")
plt.show()

#Plotting just x

for i in range(len(x)):
    plt.plot([time[i],time[i]], [0,x[i]], color = 'k', linestyle = ':')
plt.plot(time, x, '.', color = 'k')
plt.plot(time, x, color = 'r')
plt.xlim(0,20)
plt.ylim(0)
plt.xlabel("Time")
plt.ylabel(" Concentration of x Chemical")
plt.title("Brusselator - Chemical Oscillator - Just x chemical")
plt.show()

#Plotting just y

for i in range(len(y)):
    plt.plot([time[i],time[i]], [0,y[i]], color = 'k', linestyle = ':')
plt.xlabel("Time")
plt.plot(time, y, '.', color = 'k')
plt.plot(time, y, color = 'r')
plt.ylabel(" Concentration of y Chemical")
plt.title("Brusselator - Chemical Oscillator - Just y chemical")
plt.xlim(0,20)
plt.ylim(0)        


# In[ ]:




