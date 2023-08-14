#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Question 2 b)
import numpy as np
import matplotlib.pyplot as plt

# Defining all constants
Ms = 1 #Mass of sun
Mj = 0.001 #Mass of Jupiter
Mj = Mj*1000
rj = 5.2 #Jupiter's orbital radius
G = 39.5 #Gravitational constant
dt = 0.0001 #time step
t = np.arange(0,30001,1)

# Creating the position and velocity arrays for each planet and setting their inital conditions
xE = np.zeros(len(t))
xE[0] = 1
yE = np.zeros(len(t))
yE[0] = 0
v_xE = np.zeros(len(t))
v_xE[0] = 0
v_yE = np.zeros(len(t))
v_yE[0] = 6.18

xJ = np.zeros(len(t))
xJ[0] = 5.2
yJ = np.zeros(len(t))
yJ[0] = 0
v_xJ = np.zeros(len(t))
v_xJ[0] = 0
v_yJ = np.zeros(len(t))
v_yJ[0] = 2.63


# For loop runs for 3 years, since 30000 iterations at a dt = 0.0001 is 3.
for i in np.arange(0,30000):
    #Getting the Velocities for Jupiter and Earth as we've seen in question 2 a)
    v_xJ[i+1] = v_xJ[i] - ((G*xJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    v_yJ[i+1] = v_yJ[i] - ((G*yJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    r_Sun = xE[i]**2 + yE[i]**2
    r_Jup = (xE[i] - xJ[i])**2 + (yE[i] - yJ[i])**2
    v_xE[i+1] = v_xE[i] - G*dt*( ((xE[i]*Ms)/(r_Sun**(3/2))) + (((xE[i]-xJ[i])*Mj)/(r_Jup**(3/2))) )
    v_yE[i+1] = v_yE[i] - G*dt*( ((yE[i]*Ms)/(r_Sun**(3/2))) + (((yE[i]-yJ[i])*Mj)/(r_Jup**(3/2))) )
    #Updating the positions using Euler-Cromer
    xJ[i+1] = xJ[i] + v_xJ[i+1]*dt
    yJ[i+1] = yJ[i] + v_yJ[i+1]*dt
    xE[i+1] = xE[i] + v_xE[i+1]*dt
    yE[i+1] = yE[i] + v_yE[i+1]*dt

#Plots the orbit of both Jupiter and Earth for 3 years.
# A nice little check is Jupiter's full orbit takes 12 years - so for 3 years,
# Jupiter should only be 1/4 of the way around (0,0), which it is.    
plt.plot(xJ,yJ, label = "Jupiter's orbit")
plt.plot(xE,yE, label = "Earth's orbit")
plt.legend(loc = "lower right")
plt.xlabel("X position (in AU)")
plt.ylabel("Y position (in AU)")
plt.title("Orbit of Super Massive Jupiter and Earth")


# In[ ]:




