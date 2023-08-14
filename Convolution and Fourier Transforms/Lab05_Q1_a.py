#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Question 1 a)
# Author: Ilyas Sharif, Spencer Ki

# Newman Question 7.1 - using dft.py as starting point
import numpy as np
import matplotlib.pyplot as plt
from cmath import exp,pi
from numpy import zeros

def dft(y):
    N = len(y)
    c = zeros(N//2+1,complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n]*exp(-2j*pi*k*n/N)
    return c


# This is the number of points
N = 1000

# Creating our functions - making them empty arrays first
square_wave = np.empty(N, float)
sawtooth_wave = np.empty(N, float)
mod_sine_wave = np.empty(N, float)

# Creating our functions - running through the for loop.
for i in range(N):
    square_wave[i] = np.sign( np.cos((2*np.pi*i)/N) )
    sawtooth_wave[i] = i
    mod_sine_wave[i] = np.sin( (np.pi * i )/ N) * np.sin( (20 * np.pi * i )/N)

# Calculating the Discrete Fourier Transform of the functions
dft_square = abs(dft(square_wave))
dft_sawtooth = abs(dft(sawtooth_wave))
dft_mod_sine = abs(dft(mod_sine_wave))
    
    
    
# Plotting the Original functions   
plt.plot(range(N), square_wave)
plt.ylabel("$y_n$")
plt.xlabel("n")
plt.title("Square Wave Function")
plt.show()
plt.plot(range(N), sawtooth_wave)
plt.ylabel("$y_n$")
plt.xlabel("n")
plt.title("Sawtooth Wave Function")
plt.show()
plt.plot(range(N), mod_sine_wave)
plt.ylabel("$y_n$")
plt.xlabel("n")
plt.title("Modulated Sine Wave Function")
plt.show()


# Plotting the Discrete Fourier Transforms
plt.bar(range(len(dft_square)), dft_square, width = 1.2)
plt.ylabel("$\~y_n(k)$")
plt.xlabel("k")
plt.title("Fourier Transform of Square Wave")
plt.show()
plt.bar(range(len(dft_sawtooth)), dft_sawtooth, width = 1.5)
plt.ylabel("$\~y_n(k)$")
plt.xlabel("k")
plt.title("Fourier Transform of Sawtooth Wave")
plt.show()
plt.bar(range(len(dft_mod_sine)), dft_mod_sine, width = 1.7)
plt.ylabel("$\~y_n(k)$")
plt.xlabel("k")
plt.title("Fourier Transform of Modulated Sine Wave")
plt.show()


# In[ ]:




