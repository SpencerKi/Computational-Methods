# Authors: Ilyas Sharif, Spencer Ki

# Question 1 d)
# Importing essential packages
import numpy as np
import matplotlib.pyplot as plt


# Defining all inital positions, velocitys, time steps and constants (using astronomical units)
Ms = 1 #Mass of sun
G = 39.5 #Gravitational constant
dt = 0.0001 #time step
x0 = [0.47]
y0 = [0]
v_x0 = [0]
v_y0 = [8.17]
alpha = 0.01
# Creating the position and velocity arrays which we be appended to.
x = x0
y = y0
v_x = v_x0
v_y = v_y0

# Creating the Euler-Cromer function for velocity (Note: this is the same function for both and x and y dimensions just swap input places)
# x and y represents the positions, v represents the velocity
# t is time step, M is mass of the sun, G is gravitational constant
def Euler_Cromer_Velocity(x,y,v,t,M,G,a):
    Next_Velocity = v - ((G*M*x*t)/((x**2 + y**2)**(3/2)))*(1+ (a/(x**2 + y**2)))
    return Next_Velocity

# Creating the Euler-Cromer function for position (Note: this is the same function for both and x and y dimensions just swap input places)
# x represents the positions, v represents the velocity, t is the time step
def Euler_Cromer_Position(x,v,t):
    Next_Position = x + v*t
    return Next_Position

# From the pseudocode, we know we need 10000 iterations as 1 yr / dt = 10000.
# This loop will integrate for 1 year using the Euler-Cromer method to update the positions and velocities.
for i in np.arange(0,50000,1):
    v_x = np.append(v_x,Euler_Cromer_Velocity(x[i],y[i],v_x[i],dt,Ms,G,alpha))
    v_y = np.append(v_y,Euler_Cromer_Velocity(y[i],x[i],v_y[i],dt,Ms,G,alpha))
    x = np.append(x, Euler_Cromer_Position(x[i],v_x[i+1],dt))
    y = np.append(y, Euler_Cromer_Position(y[i],v_y[i+1],dt))
    
# creating time array
t = np.linspace(0,5,50001)


plt.plot(x,y)
plt.axis('equal')
plt.title("Mercury Orbit for 5 Earth Years")
plt.xlabel("X (in Au)")
plt.ylabel("Y (in Au)")

