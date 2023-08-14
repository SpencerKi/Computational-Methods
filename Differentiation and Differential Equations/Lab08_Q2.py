# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import pylab as plt

# Constants given by the question
v = 100
L = 1
d = 0.01
C = 1
sigma = 0.3

# Chosen step sizes in x and t, respectively
a = 0.001
h = 1e-6

# Velocity function given by the question
def psi_fun(x):
    return C*x*(L-x)/L**2*np.exp(-(x-d)**2/(2*sigma**2))

# Initial conditions
xpoints = np.arange(0, L + 1e-6, a)
phi = np.zeros(len(xpoints))
psi = psi_fun(xpoints)
new_phi = np.copy(phi)
new_psi = np.copy(psi)

# End time value (vary for different plots)
stop_t = 0.01

# Euler loop
t = 0
frame_counter = 0
while t <= stop_t:
    new_phi[1:-1] = phi[1:-1] + h * psi[1:-1]
    new_psi[1:-1] = psi[1:-1] + h*(v**2/a**2)*(phi[2:]+phi[:-2]-2*phi[1:-1])
    phi, psi = np.copy(new_phi), np.copy(new_psi)
    # Animate every 20 frames
#    if frame_counter % 20 == 0:
#        plt.plot(xpoints, phi)
#        plt.ylim([0,0.0001])
#        plt.draw()
#        plt.pause(0.01)
    frame_counter += 1
    
    t += h

# Final plot
plt.plot(xpoints, phi)
#plt.ylim([0,0.0002])
plt.xlabel("x (m)")
plt.ylabel("$\phi$ (m)")
plt.title("String System Time-Evolved " + str(stop_t) + " Seconds.")