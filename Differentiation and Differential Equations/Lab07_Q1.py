# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from time import time

# %% This part is adapated from the professor's Newman_8-8.py ----------------|
def rhs(r):
    M = 10.
    L = 2.

    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]

    r2 = x**2 + y**2
    Fx, Fy = - M * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25*L**2))
    return np.array([vx, Fx, vy, Fy], float)

na_N = 10000
na_h = 0.001

na_xpoints = []
na_vxpoints = []
na_ypoints = []
na_vypoints = []

na_r = np.array([1., 0., 0., 1.], float)

na_start = time()
for i in range(na_N):
    na_xpoints.append(na_r[0])
    na_vxpoints.append(na_r[1])
    na_ypoints.append(na_r[2])
    na_vypoints.append(na_r[3])
    na_k1 = na_h*rhs(na_r)
    na_k2 = na_h*rhs(na_r + 0.5*na_k1)
    na_k3 = na_h*rhs(na_r + 0.5*na_k2)
    na_k4 = na_h*rhs(na_r + na_k3)
    na_r += (na_k1 + 2*na_k2 + 2*na_k3 + na_k4)/6
na_end = time()

na_time = na_end - na_start

# %% This next part implements an adaptive timestep --------------------------|
counter = 0 # Counts required iterations
runtime = 0 # Tracks runtime
tpoints = [] # Time values per position
h = 0.01 # Initial time-step
delta = 1e-6 # Target error

# To be populated with position data
xpoints = []
vxpoints = []
ypoints = []
vypoints = []

# Initial conditions
r = np.array([1., 0., 0., 1.], float)

a_start = time() # Timing loop execution
while runtime < 10: # Loop executes for ten seconds of movement
    # One time-step of h
    k1a = h*rhs(r)
    k2a = h*rhs(r + 0.5*k1a)
    k3a = h*rhs(r + 0.5*k2a)
    k4a = h*rhs(r + k3a)
    
    # Update position based on above
    ra = r + (k1a + 2*k2a + 2*k3a + k4a)/6
    
    # Add another time-step of h
    k1a = h*rhs(ra)
    k2a = h*rhs(ra + 0.5*k1a)
    k3a = h*rhs(ra + 0.5*k2a)
    k4a = h*rhs(ra + k3a)
    
    # Again update position
    ra += (k1a + 2*k2a + 2*k3a + k4a)/6
    
    # One time-step of 2h from initial ocnditions
    k1b = 2*h*rhs(r)
    k2b = 2*h*rhs(r + 0.5*k1b)
    k3b = 2*h*rhs(r + 0.5*k2b)
    k4b = 2*h*rhs(r + k3b)
    
    # Update position based on 2h time-step
    rb = r + (k1b + 2*k2b + 2*k3b + k4b)/6
    
    # Calculate respective errors
    errx = (1/30)*(ra[0] - rb[0])
    erry = (1/30)*(ra[2] - rb[2])
    # Calculate rho based on errors
    rho = (h*delta)/np.sqrt(errx**2 + erry**2)
    
    # Adaptive step if rho is acceptable
    if rho >= 1:
        xpoints.append(r[0])
        vxpoints.append(r[1])
        ypoints.append(r[2])
        vypoints.append(r[3])
        k1 = h*rhs(r)
        k2 = h*rhs(r + 0.5*k1)
        k3 = h*rhs(r + 0.5*k2)
        k4 = h*rhs(r + k3)
        r += (k1 + 2*k2 + 2*k3 + k4)/6
        
        # Time tracking
        tpoints.append(runtime)
        runtime += h
    
    # Adapt time-step size based on rho
    h = h*rho**(1/4)
    counter += 1
# Calculate total execution time
a_end = time()
a_time = a_end - a_start

# I liked the professor's choice of font
ftsz = 16
font = {'family': 'normal', 'size': ftsz}
rc('font', **font)

# Plot adaptive, non-adaptive location overlay
plt.figure()
plt.plot(na_xpoints, na_ypoints, ':', label = 'Non-adaptive')
plt.plot(xpoints, ypoints, 'k.', label = 'Adaptive')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.title('Trajectory Calculation Method Comparison', fontsize=ftsz)
plt.axis('equal')
plt.grid()
plt.tight_layout()

# Calculate time-step size as a function of time
dtpoints = np.array(tpoints[1:]) - np.array(tpoints[:-1])
# Plot the above relationship
plt.figure()
plt.plot(tpoints[:-1], dtpoints)
plt.xlabel("Time (s)")
plt.ylabel("Time-step size (s)")
plt.title('Time-step Size vs Time', fontsize=ftsz)
plt.grid()
plt.tight_layout()