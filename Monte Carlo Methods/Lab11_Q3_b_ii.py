# -*- coding: utf-8 -*-
"""
PHY407 Lab 11, Question 3
Spencer Ki, Ilyas Sharif

Note that most of this code was adapted from Newman's salesman.py
Thank you very much for marking our labs! I hope you have a great holiday!
"""
import numpy as np
import matplotlib.pyplot as plt

# Temperature parameters
Tmax = 1e4
Tmin = 1e-3
tau = 1e4
	
# Function to minimize
def fun(x, y):
    return np.cos(x) + np.cos(np.sqrt(2)*x) + np.cos(np.sqrt(3)*x) + (y-1)**2

# Gaussian random number generator
def gauss():
    """
    Note that this function was adapted from Newman's rutherford.py
    """
    r = np.sqrt(-2*np.log(1-np.random.random()))
    theta = 2*np.pi*np.random.random()
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

# Initial conditions
x = 2
y = 2
f = fun(x,y)
t = 0
T = Tmax

# For plotting
x_holder = np.array([2,])
y_holder = np.array([2,])

# Main loop
while T>Tmin:
    
    # Cooling
    t+=1
    T = Tmax*np.exp(-t/tau)
    
    # Saving old values
    oldx = x
    oldy = y
    oldf = f
    
    # Monte Carlo step
    dx, dy = gauss()
    x += dx
    y += dy

    if x < 0 or x > 50:
        x = oldx
    if y < -20 or y > 20:
        y = oldy
    
    f = fun(x,y)
    
    delta_f = f - oldf
    if np.random.random() > np.exp(-delta_f/T):
        x = oldx
        y = oldy
    
    # For plotting
    x_holder = np.append(x_holder, x)
    y_holder = np.append(y_holder, y)

plt.figure(0)
plt.scatter(np.arange(t+1), x_holder)
plt.xlabel("Monte Carlo step")
plt.ylabel("x")
plt.title("Minimization Process via Simulated Annealing")

plt.figure(1)
plt.scatter(np.arange(t+1), y_holder)
plt.xlabel("Monte Carlo step")
plt.ylabel("y")
plt.title("Minimization Process via Simulated Annealing")

print(x)
print(y)
print(f)