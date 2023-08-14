# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import pylab as plt

phi = np.zeros([81, 201], float)  # Generate grid
phi[80,:51] = np.linspace(0,5,51) # AB
phi[50:,50] = np.linspace(7,5,31) # BC
phi[50,50:151] = 7                # CD
phi[50:,150] = np.linspace(7,5,31)# DE
phi[80,150:] = np.linspace(5,0,51)# EF
phi[:,200] = np.linspace(10,0,81) # FG
phi[0,:] = 10                     # GH
phi[:,0] = np.linspace(10,0,81)   # HA

target = 1e-6 # Target accuracy
omega = 0.9   # Overrelaxation parameter (set to 0 for Q1a and some of Q1b)
delta = 1     # Initial delta value

# %% The following adapted the Jacobi method from Newman's laplace.py to apply
# the Gauss-Seidel method with animation. ------------------------------------\

while delta>target: # Modified to 100 iterations for Q1b
    placeholder = np.copy(phi) # Copy of 'current' phi to calculate precision
    
    for i in range(1,50):
        for j in range(1,200):
            phi[i,j] =\
            (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])*(1 + omega)/4 \
            - omega*phi[i,j]
    for i in range(50,80):
        for j in range(1,50):
            phi[i,j] =\
            (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])*(1 + omega)/4 \
            - omega*phi[i,j]
    for i in range(50,80):
        for j in range(151,200):
            phi[i,j] =\
            (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])*(1 + omega)/4 \
            - omega*phi[i,j]
    # Animation for each iteration
    plt.imshow(phi)
    plt.hot()
    plt.draw()
    plt.pause(0.01)
    
    delta = np.amax(abs(phi-placeholder))
    
# Label plot
plt.imshow(phi)
plt.hot()
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Temperature Distribution (C), $\omega$ = 0.9")
plt.colorbar()

# Calculate temp. at (2.5, 1)
print(phi[70,25])