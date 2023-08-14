# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import scipy.constants as cn
import pylab as plt

# Constants given by the question
L = 1e-8
sigma = L/25
k = 500/L

# Step sizes and spacings in x and t, respectively
P = 1024
N = 4000
delta = L/P
tau = 1e-18

# Start and end point
x_0 = L/5
x_1 = L/4

# Normalizing using Simpson's method for integration
def norm(x):
    return np.exp(-(x-x_0)**2/(2*sigma**2))
Simpson = norm(-L/2) + norm(L/2)
for i in range(1,P,2):
    Simpson += norm(-L/2+delta)*4
for i in range(2,P,2):
    Simpson += norm(-L/2+delta)*2
Simpson *= delta/3
psi_0 = np.sqrt(1/Simpson)

# Potential for this scenario
def V(x):
    return 0.5*cn.electron_mass*3e15**2*x**2

# Constructing the Hamiltonian
A = -cn.hbar**2/(2*cn.electron_mass*delta)
def B(p):
    V(p*delta-L/2)-2*A

hamiltonian = np.zeros((P-1,P-1))
hamiltonian[0][0] = B(1)
hamiltonian[0][1] = A
hamiltonian[-1][-1] = B(P)
hamiltonian[-1][-2] = A
for i in range(2, P - 1):
    hamiltonian[i-1][i-1] = B(i)
    hamiltonian[i-1][i-2] = A
    hamiltonian[i-1][i] = A

# Linear system of equations from lab manual
def next_psi(last_psi):
    L = np.identity(P - 1) + 1j*(tau/(2*cn.hbar))*hamiltonian
    R = np.identity(P - 1) - 1j*(tau/(2*cn.hbar))*hamiltonian    
    return np.linalg.solve(L, np.dot(R,last_psi))

# Initial conditions
def psi_fun(x):
    return psi_0*np.exp(-(x-x_0)**2/(4*sigma**2)+1j*k*x)
x_points = delta* np.arange(1,P)-L/2
t_points = np.arange(0,N*tau,tau)
psi = psi_fun(x_points)

# Iteration loop
placeholder = psi.copy() 
wave_fun = [psi.copy()]
for i in t_points[1:]:
    new_psi = next_psi(placeholder)
    wave_fun.append(new_psi)
    placeholder = new_psi.copy()

# Plotting
position_0 = np.conjugate(wave_fun[0])*wave_fun[0]
position_1 = np.conjugate(wave_fun[int(N*tau/4)])*wave_fun[int(N*tau/4)]
position_2 = np.conjugate(wave_fun[int(N*tau/2)])*wave_fun[int(N*tau/2)]
position_3 = np.conjugate(wave_fun[int(N*tau)])*wave_fun[int(N*tau)]

plt.figure(0)
plt.plot(x_points, abs(position_0/5), label = 't = 0')
plt.plot(x_points, abs(position_1/5), label = 't = T/4')
plt.plot(x_points, abs(position_2/5), label = 't = T/2')
plt.plot(x_points, abs(position_3/5), label = 't = T')
plt.xlabel('Position (m)')
plt.ylabel('Probability')
plt.legend()
plt.title('Probability of Electron Position')