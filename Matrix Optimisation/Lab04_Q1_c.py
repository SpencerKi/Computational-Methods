# -*- coding: utf-8 -*-
"""
Authors: Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from SolveLinear import GaussElim, PartialPivot

# Constants
R1 = 1e3
R2 = 2e3
R3 = 1e3
R4 = 2e3
R5 = 1e3
R6 = 2e3
C1 = 1e-6
C2 = 0.5e-6
xp = 3
w = 1000

# Circuit model
circuit = np.array([[1/R1+1/R4+1j*w*C1, -1j*w*C1, 0],\
                    [-1j*w*C1, 1/R2+1/R5+1j*w*C1+1j*w*C2, -1j*w*C2],\
                    [0, -1j*w*C2, 1/R3+1/R6+1j*w*C2]], dtype = complex)
circuit_vector = np.array([xp/R1, xp/R2, xp/R3], dtype = complex)

# Circuit solutions and characteristics
circuit_sol = PartialPivot(circuit, circuit_vector)

circuit_ampls = np.abs(circuit_sol)
circuit_phases = np.angle(circuit_sol)

# Voltage function
def voltage(var, shift, t):
    return var*np.exp(1j*(w*t+shift), dtype = complex)

# Plot
plt.figure(0)
t = np.linspace(0, 16)
plt.plot(t, voltage(circuit_sol[0], circuit_phases[0], t), label = "V1")
plt.plot(t, voltage(circuit_sol[1], circuit_phases[1], t), label = "V2")
plt.plot(t, voltage(circuit_sol[2], circuit_phases[2], t), label = "V3")
plt.legend()
plt.title("Voltage at Three Junctions")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")