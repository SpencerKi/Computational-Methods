# Authors: Spencer Ki, Ilyas Sharif

#Question 3
import numpy as np
from scipy import constants as con
from scipy import special as sp
import matplotlib.pyplot as plt
from matplotlib import cm

# This technically isn't required, but for some reason the code will not run
# without it.
from mpl_toolkits import mplot3d 

# Values given in question.
Q = 10e-13
l = 1e-3

# Equation 8 for calculating electric potential (without integral).
def elec_pot8(r, u, z):
    return (Q * np.exp(-np.tan(u)**2)) / (4*con.pi*con.epsilon_0*np.cos(u)**2*\
    np.sqrt((z - l*np.tan(u))**2 + r**2))

# Equation 9 for calculating electric potential.
def elec_pot9(r):
    return Q / (4*con.pi*con.epsilon_0*l) * np.exp(r**2/(2*l**2))* sp.kn(0, r**2/(2*l**2))

# Simpson's rule applied to integrating equation 8.
def simpson(r, N, z):
    a = -con.pi/2 #starting point
    b = con.pi/2 #end point
    h = (b-a)/N #width of slice
    
    # Simpson Rules - find inital points, do 2 sums, one for even, one for odd
    placeholder = elec_pot8(r, a, z) + elec_pot8(r, b, z) #inital points
    for k in range(1,N,2): #the odd sum in simpson's rule
        placeholder += elec_pot8(r, a+k*h, z)*4
    for k in range(2,N,2): #the even sum in simpson's rule
        placeholder += elec_pot8(r, a+k*h, z)*2
    #must multiply by h/3 to get proper value of integral
    return (h/3)*placeholder

# Initialise variable since this lengthiis used repeatedly.
length = np.linspace(0.25e-3, 5e-3,1000)

# Plot results
plt.figure(0)
plt.plot(length, simpson(length, 8, 0), label = "Simpson's Rule Calculation")
plt.plot(length, elec_pot9(length), label = "Scipy Calculation")
plt.legend()
plt.title("Electrostatic Potential Calculation: Simpson's Rule vs Scipy")
plt.xlabel("Radius (m)")
plt.ylabel("Potential (V)")

plt.figure(1)
plt.plot(length, elec_pot9(length)- simpson(length, 8, 0))
plt.title("Difference Between Electrostatic Potential Calculation Methods")
plt.xlabel("Radius (m)")
plt.ylabel("Arithmetic Difference")

plt.figure(2)
plt.plot(length, elec_pot9(length)- simpson(length, 66, 0))
plt.title("Difference Between Calculation Methods, N = 66")
plt.xlabel("Radius (m)")
plt.ylabel("Arithmetic Difference")

# 3D plot code modified from online resource, as instructed.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = np.linspace(-5e-3, 5e-3,1000)
Y = np.linspace(-5e-3, 5e-3,1000)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = simpson(length, 66, R)

plt.xlabel("Distance from Origin (m)")
plt.ylabel("Distance from Origin (m)")

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5, label = "Potential (V)")

plt.title("Potential Field Calculated Via Simpson's Rule")
plt.show()