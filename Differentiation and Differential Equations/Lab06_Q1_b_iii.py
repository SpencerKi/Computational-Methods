#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Question 1 b) iii
# Author: Ilyas Sharif

import numpy as np
import matplotlib.pyplot as plt


# Defining constants - as defined in the handout
v_f = 0.1
omega_0 = 1
tau = 1
gamma = 0.5
# v_p < gamma * tau (from question 1a iii.)
# v_p as derived from part v)
v_p = v_f * np.log((gamma*tau)/v_f)


# Following Newman's code from Chapter 8.3
# You can essentially define a higher order ODE (nonlinear or linear)
# as a simultaneous first order ODE. This is what we do here
# we define dx/dt = y and then our second order ODE becomes a first order ODE
# with respect to y. We can then create an array for r = (x, y)
# and solve these functions.

# Creating f(r,t) where r = (x, y = dx/dt)
def f(r,t):
    x = r[0]
    y = r[1]
    fx = y
    fy = -(omega_0**2)*((x) - v_p * t) - (y)/tau - gamma*np.exp(-np.abs(y)/v_f)
    return np.array([fx, fy], float)

# Copying code from Newman odesim.py
a = 0.0
b = 100.0
N = 10000
h = (b-a)/N

# Defining the time, x and y arrays.
tpoints = np.arange(a, b,h)
xpoints = []
ypoints = []


# Creating inital positions and velocities (Initial velocity is arbitrary for this example).
C = v_p
x_0 = -(1/omega_0**2)*(C/tau + gamma*np.exp(-C/v_f))
y_0 = 5


# Creating r array and computing RK4 method (copied from Newman odesim.py)
r = np.array([x_0, y_0], float) 
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6

    
# Constant x where velocity becomes constant (i.e. solution from Q1(a)iv.)
x0 = -(1/omega_0**2)*(C/tau + gamma*np.exp(-C/v_f)) + (v_p*tpoints)


# Plotting
plt.plot(tpoints, xpoints, label = "Simple Harmonic Solution (Q1(a)v)")
plt.plot(tpoints,x0, linestyle = ":", color = 'k' , label = "Constant-Velocity Solution (Q1(a)iv)")
plt.xlim(0,25)
plt.legend()
plt.title("Comparison of Simple Harmonic Motion and Constant-Velocity Solution")
plt.xlabel(" time (seconds) ")
plt.ylabel(" position (meters) ")
plt.show()


# In[ ]:




