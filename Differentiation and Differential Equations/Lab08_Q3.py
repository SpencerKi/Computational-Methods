#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 3
# Author: Ilyas Sharif

# importing required modules
import numpy as np
import matplotlib.pyplot as plt

# Defining the parameters as given in the lab manual
epsilon = 1
delta_x = 0.02
delta_t = 0.005
Lx = 2 * np.pi
Tf = 2
Nx = int(Lx / delta_x)

# Defining beta, from equation 9 of computational background
# As instructed in computational background, eq. 6, 7
Beta = epsilon* ( delta_t / delta_x )

# Creating x and t arrays from inital parameters
x = np.arange(0, Lx, delta_x)
t = np.arange(0, Tf, delta_t)

# Creating u_i^j as defined in computational background
u = np.zeros([len(x), len(t)])

# Defining our inital conditions
# Range given from eq. 6, 7 in computational background
for i in range(1, Nx-1):
    # u(x, t = 0) = sin(x) as desired initial condition
    u[i,0] = np.sin(x[i])
    # mandatory required time step before applying equation 9
    # derived using Euler i.e. u(x,∆t) = u(x,0) + d/dx(u(x, t=0)*∆t
    u[i,1] = u[i,0] + np.cos(x[i])*delta_t
    
# From the lab handout:
# The space and time directions are discretized into Nx and Nt 
# steps of sizes ∆x and ∆t. from eq. 6,7 we know we range to Nx - 1 and Nt - 1
# we can ignore Nt for calculating the range for equation 9.
for j in range(1, Nx-1):
    for i in range(1, Nx-1):
        u[0, i] = 0 # Boundary condition u(0, t) = 0
        u[-1, i] = 0 # Boundary condition u(Lx, t) = 0
        
        # Equation 9 from computational background
        u[i, j+1] = u[i, j-1] - (Beta / 2) * ( (u[i+1,j])**2 - (u[i-1,j])**2)
        

# Plotting u(x, t) for t = 0, 0.5, 1, 1.5
# numpy's nifty function np.where finds the index of time
# time array where the t = desired value
# we then output the integer value of this index
# to get the correct plot for u(x, t = desired time)


# u(x, t = 0)
plt.plot(x, u[:,int((np.where(t==0))[0])], color = 'k')
plt.xlabel("x")
plt.ylabel("u(x, t = 0)")
plt.title(" Burger's Equation for u(x, t = 0)")
plt.xlim(x[0],x[len(x)-1])
plt.show()

# u(x, t = 0.5)
plt.plot(x, u[:,int((np.where(t==0.5))[0])], color = 'k')
plt.xlabel("x")
plt.ylabel("u(x, t = 0.5)")
plt.title(" Burger's Equation for u(x, t = 0.5)")
plt.xlim(x[0],x[len(x)-1])
plt.show()

# u(x, t = 1)
plt.plot(x, u[:,int((np.where(t==1))[0])], color = 'k')
plt.xlabel("x")
plt.ylabel("u(x, t = 1)")
plt.title(" Burger's Equation for u(x, t = 1)")
plt.xlim(x[0],x[len(x)-1])
plt.show()

# u(x, t = 1.5)
plt.plot(x, u[:,int((np.where(t==1.55))[0])], color = 'k')
plt.xlabel("x")
plt.ylabel("u(x, t = 1.5)")
plt.title(" Burger's Equation for u(x, t = 1.5)")
plt.xlim(x[0],x[len(x)-1])
plt.show()


# In[ ]:




