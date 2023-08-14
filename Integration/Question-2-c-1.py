#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Question 2 c)
import numpy as np
import matplotlib.pyplot as plt

# Defining all constants
Ms = 1 #Mass of sun
Mj = 0.001 #Mass of Jupiter
rj = 5.2 #Jupiter's orbital radius
G = 39.5 #Gravitational constant
dt = 0.0001 #time step
t = np.arange(0,2000001,1)

# Creating the position and velocity arrays for each planet and setting their inital conditions
xE = np.zeros(len(t))
xE[0] = 3.3
yE = np.zeros(len(t))
yE[0] = 0
v_xE = np.zeros(len(t))
v_xE[0] = 0
v_yE = np.zeros(len(t))
v_yE[0] = 3.46

xJ = np.zeros(len(t))
xJ[0] = 5.2
yJ = np.zeros(len(t))
yJ[0] = 0
v_xJ = np.zeros(len(t))
v_xJ[0] = 0
v_yJ = np.zeros(len(t))
v_yJ[0] = 2.63

#Running the for loop for 200000 iterations, which for a time step of 0.0001 is 20 years.
for i in np.arange(0,2000000,1):
    #Gets the velocity of Jupiter and Earth as seen before in Question 2 a) and b)
    v_xJ[i+1] = v_xJ[i] - ((G*xJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    v_yJ[i+1] = v_yJ[i] - ((G*yJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    r_Sun = xE[i]**2 + yE[i]**2
    r_Jup = (xE[i] - xJ[i])**2 + (yE[i] - yJ[i])**2
    v_xE[i+1] = v_xE[i] - G*dt*( ((xE[i]*Ms)/(r_Sun**(3/2))) + (((xE[i]-xJ[i])*Mj)/(r_Jup**(3/2))) )
    v_yE[i+1] = v_yE[i] - G*dt*( ((yE[i]*Ms)/(r_Sun**(3/2))) + (((yE[i]-yJ[i])*Mj)/(r_Jup**(3/2))) )
    #Gets the positions using Euler-Cromer - nothing new here.
    xJ[i+1] = xJ[i] + v_xJ[i+1]*dt
    yJ[i+1] = yJ[i] + v_yJ[i+1]*dt
    xE[i+1] = xE[i] + v_xE[i+1]*dt
    yE[i+1] = yE[i] + v_yE[i+1]*dt

#plots the orbit of the asteroid and Jupiter     
plt.plot(xJ,yJ, label = "Jupiter's orbit")
plt.plot(xE,yE, label = "Asteroid's orbit")
plt.legend(loc = 1)
plt.xlabel("X position (in AU)")
plt.ylabel("Y position (in AU)")
plt.title("Orbit of Jupiter and an Asteroid")


# In[ ]:




