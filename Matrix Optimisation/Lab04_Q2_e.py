#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2 - part e
# Author: Ilyas Sharif, Spencer Ki

import numpy as np
from numpy import pi, sin
from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as plt


# Defining all the inital constants as provided by Newmann.
# Had to look up the hbar value from wikipedia though.
L = 5e-10 # width in m
a = 10 # eV
M = 9.1094e-31 # mass of electron in kg
charge = 1.6022e-19 # Charge of electron in coloumbs.
hbar = 6.58211e-16 # h-bar in eV * seconds


# Defining our H matrix function
def Hmatrix(m, n):
    if m == n:
        return (a/2) + ( ((pi**2)*(hbar**2)*(m**2))/ (2*M*(L**2)))*charge
    elif not m % 2 == n % 2:
        return -(8*a*m*n)/( (pi**2) * ((m**2 - n**2)**2)  )
    else:
        return 0

    
# Defining the maxmimum n and m values, creating the empty H matrix. This helps with indexing    
mmax = 100    
nmax = mmax
H = np.empty([ mmax, nmax ])

# Creating our values for the H matrix.
for m in range(1, mmax+1):
    for n in range(1, nmax+1):
        H[m-1, n-1] = Hmatrix(m, n)
    
# Using Numpy to calculate eigenvalues 
Energy_eigenvals, eigenvectors = eigh(H)

# Creating the function to calculate our wavefunctions
# input is the x axis and n is the energy level, n=0 is ground state, n =1  is first excited, etc.
def wavefunction(x, n):
    wavefunc = 0
    for m in range(nmax):
        wavefunc += eigenvectors[m][n] * sin((pi*(m+1)*x)/L)
    return wavefunc


# creating x array
x = np.linspace(0, L, 100)
# Terms for simpson integration
N = 50
a = 0
b = L
h = (b-a)/N

# Running the Integration - Using Simpson's rule
# Going to make it a function so i can test the values of my normalized wave functions
# Simpson Rules - find inital points, do 2 sums, one for even, one for odd
def norm_wave_squared_integral(normalization, n):
    Simpson = np.abs(wavefunction(a, n) / normalization)**2 + np.abs(wavefunction(b, n) / normalization)**2 #inital points
    for k in range(1,N,2): #the odd sum in simpson's rule
        Simpson += ( np.abs(wavefunction(a+k*h, n) / normalization)**2 )*4
    for k in range(2,N,2): #the even sum in simpson's rule
        Simpson += (np.abs(wavefunction(a+k*h, n) / normalization)**2 ) *2
    #must multiply by h/3 to get proper value of integral
    return  (h/3)*Simpson

# Creating our normalization array / normalization for each energy level
normalization = []
for i in range(3):
    n = i
    norm = np.sqrt(norm_wave_squared_integral(1,n))
    normalization.append(norm)
        
    
#creating our NORMALIZED wavefunctions
wave_ground = wavefunction(x,0) / normalization[0]
wave_ground = np.abs(wave_ground)**2
wave_first = wavefunction(x,1) / normalization[1]
wave_first = np.abs(wave_first)**2
wave_second = wavefunction(x,2) / normalization[2]
wave_second = np.abs(wave_second)**2


# Plotting the norm squared of our normalized wavefunctions.
plt.title("Normalized Wavefunctions")
plt.ylabel("|$\Psi$(x)| $^2$")
plt.xlabel("x")
plt.xlim(0,5e-10)
plt.plot(x,wave_ground, label = 'Ground State')
plt.plot(x,wave_first, label = 'First Excited State')
plt.plot(x,wave_second, label = 'Second Excited State')
plt.legend()
plt.show()
plt.title("Unnormalized Wavefunctions")
plt.plot(x,np.abs(wavefunction(x,0))**2, label = 'Ground State')
plt.plot(x,np.abs(wavefunction(x,1))**2, label = 'First Excited State')
plt.plot(x,np.abs(wavefunction(x,2))**2, label = 'Second Excited State')
plt.ylabel("|$\Psi$(x)| $^2$")
plt.xlabel("x")
plt.xlim(0,5e-10)
plt.show()

# Just to prove our wavefunctions are normalized, we will calculate the integral
# of each wavefunction divided by sqrt A, from 0 to L. This should be equal to 1
for i in range(3):
    n = i
    normalized_wavefunction_integral = norm_wave_squared_integral(normalization[i],n)
    print(" The integral of the norm of the wavefunction squared from 0 to L for n = ", i , " is ", normalized_wavefunction_integral)
    


# In[ ]:




