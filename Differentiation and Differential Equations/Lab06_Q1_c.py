#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Question 1 c)
# Author: Ilyas Sharif

import numpy as np
import matplotlib.pyplot as plt


# Defining the parameters that didn't change (same as code for before)
v_f = 0.1
omega_0 = 1
tau = 1
gamma = 0.5
a = 0.0
b = 100.0
N = 10000
h = (b-a)/N
tpoints = np.arange(a, b,h)
x_0 = 0
y_0 = 0



# Defining the xpoints and ypoints array.
vp = v_f * np.log((gamma*tau)/v_f)
v_p = [0.1*vp,0.25*vp, 0.5*vp, 0.75*vp, 1*vp, 1.25*vp, 1.5*vp]
indexing = [0.1,0.25,0.5,0.75,1,1.25,1.5]

for i in range(len(v_p)):
    C = v_p[i]
    xpoints = []
    r = np.array([x_0, y_0], float) 
    # Creating f(r,t) where r = (x, y = dx/dt)
    def f(r,t):
        x = r[0]
        y = r[1]
        fx = y
        fy = -(omega_0**2)*((x) - v_p[i] * t) - (y)/tau - gamma*np.exp(-np.abs(y)/v_f)
        return np.array([fx, fy], float)


    # Creating r array and computing RK4 method (copied from Newman odesim.py)
    for t in tpoints:
        xpoints.append(r[0])
        k1 = h*f(r,t)
        k2 = h*f(r+0.5*k1,t+0.5*h)
        k3 = h*f(r+0.5*k2,t+0.5*h)
        k4 = h*f(r+k3,t+h)
        r += (k1+2*k2+2*k3+k4)/6
    plt.plot(tpoints, xpoints, label = '$v_p$ = ' + str(indexing[i]) + '$v_p$')
    # I'm going to comment this out, but if you want to see the constant velocity
    # solutions that each of them oscillate around, feel free to comment out the
    # 2 lines below :)
    #x0 = -(1/omega_0**2)*(C/tau + gamma*np.exp(-C/v_f)) + (v_p[i]*tpoints)
    #plt.plot(tpoints,x0, linestyle = ":", color = 'k')

    

plt.title("Comparison of Different Choices for $v_p$")
plt.xlabel(" time (seconds) ")
plt.ylabel(" position (meters) ")
plt.legend()
plt.xlim(0,100)
plt.show()


# In[ ]:




