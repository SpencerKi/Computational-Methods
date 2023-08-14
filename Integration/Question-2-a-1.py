#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2 a)
import numpy as np
import matplotlib.pyplot as plt

# Defining all constants
Ms = 1 #Mass of sun
Mj = 0.001 #Mass of Jupiter
rj = 5.2 #Jupiter's orbital radius
G = 39.5 #Gravitational constant
dt = 0.0001 #time step
t = np.arange(0,100001,1)

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

#Running a for loop for 10 years with the time step dt = 0.0001
for i in np.arange(0,100000,1):
    #This updates the velocities of Jupiter using Euler-Cromer method from Question 1
    v_xJ[i+1] = v_xJ[i] - ((G*xJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    v_yJ[i+1] = v_yJ[i] - ((G*yJ[i]*dt*Ms)/((xJ[i]**2 + yJ[i]**2)**(3/2)))
    #r_Sun and r_Jup just make the code neater
    r_Sun = xE[i]**2 + yE[i]**2
    r_Jup = (xJ[i] - xE[i])**2 + (yJ[i] - yE[i])**2
    #By solving the gravitational forces equations for Jupiter Sun, you get this result.
    v_xE[i+1] = v_xE[i] - G*dt*( ((xE[i]*Ms)/(r_Sun**(3/2))) - (((xJ[i]-xE[i])*Mj)/(r_Jup**(3/2))) )
    v_yE[i+1] = v_yE[i] - G*dt*( ((yE[i]*Ms)/(r_Sun**(3/2))) - (((yJ[i]-yE[i])*Mj)/(r_Jup**(3/2))) )
    
    #Updates the positions of Earth and Jupiter - must be done after the velocities update.
    xJ[i+1] = xJ[i] + v_xJ[i+1]*dt
    yJ[i+1] = yJ[i] + v_yJ[i+1]*dt
    xE[i+1] = xE[i] + v_xE[i+1]*dt
    yE[i+1] = yE[i] + v_yE[i+1]*dt


#Plotting Earth and Jupiter's orbit together    
plt.plot(xJ,yJ, label = "Jupiter's orbit")
plt.plot(xE,yE, label = "Earth's orbit")
plt.legend(loc = "center right")
plt.xlabel("X position (in AU)")
plt.ylabel("Y position (in AU)")
plt.title("Orbit of Jupiter and Earth")


# In[ ]:




