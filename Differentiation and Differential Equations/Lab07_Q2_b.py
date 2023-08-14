#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2 b)
# Author: Ilyas Sharif

import numpy as np
from numpy import array,arange

# NOTE: this code is heavily based off example 8.9 from textbook (squarewell.py)

# Constants
m = 9.1094e-31 # Mass of electron
hbar = 1.0546e-34 # Planck's constant over 2*pi
e = 1.6022e-19 # Electron charge
a = 10e-12 # potential constant
V0 = 50*e # potential constant
L = (20*a) # large finite interval
N = 1000 # Number of Steps in Runge-Kutta
h = L/N

# Potential function
def V(x):
    return (V0*(x**4))/(a**4)

def f(r,x,E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/(hbar**2))*(V(x)-E)*psi
    return array([fpsi,fphi] ,float)

# Calculate the wavefunction for a particular energy
def solve(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi] ,float)
    
    for x in arange(-10*a,10*a,h):
        k1 = h*f(r,x,E)
        k2 = h*f(r+0.5*k1,x+0.5*h,E)
        k3 = h*f(r+0.5*k2,x+0.5*h,E)
        k4 = h*f(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
        
    return r[0]

# Main program to find the energy using the secant method
def secant(E1, E2):
    psi2 = solve(E1)

    target = e/1000
    while abs(E1-E2)>target:
        psi1,psi2 = psi2,solve(E2)
        E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
        
    return E2

Ground_state = secant(0, e) / e
First_excited = secant(400*e, 401*e) / e
Second_excited = secant(401*e, 1000*e) / e
    
print("E =" , Ground_state, "eV")
print("E =" , First_excited, "eV")
print("E =" , Second_excited, "eV")


# In[ ]:




