# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants given by question
km = 400
dt = 1e-3

# Define function for solving ODE system via Verlet
def vib(N, steps):
    
# Generating NxN 'A' matrix
    A = np.zeros([N,N])
    A[0][0] = -2
    A[0][1] = 1
    A[-1][-1] = -2
    A[-1][-2] = 1
    for i in range(1, N-1):
        A[i][i] = -2
        A[i][i+1] = 1
        A[i][i-1] = 1
            
# Initialising displacement and half-step velocity arrays
    d = np.zeros((2,N))
    d[1][0] = 0.1
    hv = np.zeros(N)
    hv = np.vstack((hv, 0.5*dt*km*np.matmul(A, d[-1])))
    
# Adapted Verlet method
    for i in range(steps):
        d = np.vstack((d, d[-1] + dt * hv[-1]))
        k = dt * km * np.matmul(A, d[-1])
        hv = np.vstack((hv, hv[-1] + k))
    
    return d[1:]
    
# Plotting N = 3,10 in the short- and the long-term
plt.figure(0)
plt.plot(dt*np.arange(1001), vib(3, 1000))
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 3-Storey Building over 1s")

plt.figure(1)
plt.plot(dt*np.arange(10001), vib(3, 10000))
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 3-Storey Building over 10s")

plt.figure(2)
plt.plot(dt*np.arange(1001), vib(10, 1000))
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 10-Storey Building over 1s")

plt.figure(3)
plt.plot(dt*np.arange(10001), vib(10, 10000))
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 10-Storey Building over 10s")

# N = 3 'A' matrix for Q2b
three = km*np.array([[-2.,  1.,  0.], [ 1., -2.,  1.], [ 0.,  1., -2.]])
# Calculating eigenvalues and eigenvectors of 'A'
ei = np.linalg.eigh(three)

# Normal modes function
def oscil(amp, freq, t):
    return amp*np.cos(freq*t)

# Plotting normal modes
plt.figure(4)
plt.plot(dt*np.arange(1000), oscil(ei[1][0][0], np.sqrt(-ei[0][0]), dt*np.arange(1000)), label = 'Floor 0')
plt.plot(dt*np.arange(1000), oscil(ei[1][1][0], np.sqrt(-ei[0][0]), dt*np.arange(1000)), label = 'Floor 1')
plt.plot(dt*np.arange(1000), oscil(ei[1][2][0], np.sqrt(-ei[0][0]), dt*np.arange(1000)), label = 'Floor 2')
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.legend()
plt.title("Oscillation of 3-Storey Building's Normal Mode 0 over 1s")

plt.figure(5)
plt.plot(dt*np.arange(1000), oscil(ei[1][0][1], np.sqrt(-ei[0][1]), dt*np.arange(1000)), label = 'Floor 0')
plt.plot(dt*np.arange(1000), oscil(ei[1][1][1], np.sqrt(-ei[0][1]), dt*np.arange(1000)), label = 'Floor 1')
plt.plot(dt*np.arange(1000), oscil(ei[1][2][1], np.sqrt(-ei[0][1]), dt*np.arange(1000)), label = 'Floor 2')
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.legend()
plt.title("Oscillation of 3-Storey Building's Normal Mode 1 over 1s")

plt.figure(6)
plt.plot(dt*np.arange(1000), oscil(ei[1][0][2], np.sqrt(-ei[0][2]), dt*np.arange(1000)), label = 'Floor 0')
plt.plot(dt*np.arange(1000), oscil(ei[1][1][2], np.sqrt(-ei[0][2]), dt*np.arange(1000)), label = 'Floor 1')
plt.plot(dt*np.arange(1000), oscil(ei[1][2][2], np.sqrt(-ei[0][2]), dt*np.arange(1000)), label = 'Floor 2')
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.legend()
plt.title("Oscillation of 3-Storey Building's Normal Mode 2 over 1s")

# Comparing eigenvalue analysis to simulation
zer = np.zeros(3)
d0 = np.vstack((zer, np.transpose(ei[1])[0]))
d1 = np.vstack((zer, np.transpose(ei[1])[1]))
d2 = np.vstack((zer, np.transpose(ei[1])[2]))
hv0 = np.vstack((zer, 0.5*dt*np.matmul(three, d0[-1])))
hv1 = np.vstack((zer, 0.5*dt*np.matmul(three, d1[-1])))
hv2 = np.vstack((zer, 0.5*dt*np.matmul(three, d2[-1])))
for j in range(1000):
    d0 = np.vstack((d0, d0[-1] + dt * hv0[-1]))
    d1 = np.vstack((d1, d1[-1] + dt * hv1[-1]))
    d2 = np.vstack((d2, d2[-1] + dt * hv2[-1]))
    k0 = dt * np.matmul(three, d0[-1])
    k1 = dt * np.matmul(three, d1[-1])
    k2 = dt * np.matmul(three, d2[-1])
    hv0 = np.vstack((hv0, hv0[-1] + k0))
    hv1 = np.vstack((hv1, hv1[-1] + k1))
    hv2 = np.vstack((hv2, hv2[-1] + k2))
d0 = d0[1:]
d1 = d1[1:]
d2 = d2[1:]

# Plotting comparison
plt.figure(7)
plt.plot(dt*np.arange(1001), d0)
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 3-Storey Building over 1s [Eigenvector]")

plt.figure(8)
plt.plot(dt*np.arange(1001), d1)
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 3-Storey Building over 1s [Eigenvector]")

plt.figure(9)
plt.plot(dt*np.arange(1001), d2)
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Displacement (m)")
plt.title("Motion of Vibrating 3-Storey Building over 1s [Eigenvector]")