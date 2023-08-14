#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Question 4
# Author: Ilyas Sharif

# import modules
import numpy as np
import matplotlib.pyplot as plt
from random import random, randrange
from pylab import *


# Function to calculate energy
def energyfunction(J_, dipoles):
    # Will go through every row and column to get the adjacent interactions
    # Then will sum together at the end.
    # Note: super similar to Ising1D, just applied to another axis ("dimension")
    
    energy_row = -J_ * sum( (dipoles[:-1, :] * dipoles[1:, :]))
    energy_column = -J_ * sum( (dipoles[:, :-1] * dipoles[:, 1:]))

    energy = energy_row + energy_column
    return energy

# Function for accepetance probability
def acceptance(Enew, E, kB, T):
    result = False #by default, reject new state
    DeltaE = Enew - E
    p = np.exp(-(DeltaE/(kB*T)))
    if DeltaE <= 0.0:
        result = True
    elif DeltaE > 0 and p > random():
        result = True
    return result

## Define constants
kB = 1
T = 1 #Change this value to 2, 3 for part (e)
J = 1
N = 100000 # Number of flips
num_dipoles = 20 # Number of dipoles per side, i.e. 20x20

# Creating initial array for system of 20x20 spins
# where each spin can be either +1 or -1 (randomly)

dipoles = np.ones([num_dipoles, num_dipoles], int)

for i in range(num_dipoles):
    for j in range(num_dipoles):
        flip = random()
        if flip < 0.5:
            dipoles[i][j] = -1

# Plot the initial dipole
plt.imshow(dipoles)
plt.xticks(range(0,20,2), range(1,21,2))
plt.yticks(range(0,20,2), range(1,21,2))     
plt.title("Initial Spin State")
plt.colorbar()
plt.show()
            
# generate array of dipoles and initialize diagnostic quantities
energy = []  # empty list; to add to it, use energy.append(value)
magnet = []  # empty list; to add to it, use magnet.append(value)
# Get initial values for magnetization and Energy        
E = energyfunction(J, dipoles)
energy.append(E)
magnet.append(np.sum(dipoles))

for i in range(N):
    # choose a victim
    picked_row = randrange(num_dipoles)  
    picked_column = randrange(num_dipoles)
    
    # propose to flip the victim
    dipoles[picked_row][picked_column] *= -1
    
    # compute Energy of proposed new state
    Enew = energyfunction(J, dipoles)

    # calculate acceptance probability
    accepted = acceptance(Enew, E, kB, T)

    # store energy and magnetization
    # If we accept the new value, append the values to
    # energy and magnet. Else, flip back and try again
    # with a new position.
    
    if accepted:
        Mnew = np.sum(dipoles)
        E = Enew
    else:
        dipoles[picked_row][picked_column] *= -1 # flip back
        Mnew = np.sum(dipoles)
            
    magnet.append(Mnew)
    energy.append(E)
    
    if i == N//2:
        # Plot the dipole halfway through
        plt.imshow(dipoles)
        plt.xticks(range(0,20,2), range(1,21,2))
        plt.yticks(range(0,20,2), range(1,21,2))     
        plt.title("Intermediate Spin State")
        plt.colorbar()
        plt.show()        
    
    
# Plot the final dipole
plt.imshow(dipoles)
plt.xticks(range(0,20,2), range(1,21,2))
plt.yticks(range(0,20,2), range(1,21,2))     
plt.title("Final Spin State")
plt.colorbar()
plt.show()    
    
# make plots of magnetization and energy
fg, ax = plt.subplots(2, 1, sharex = True)
ax[0].plot(magnet)
ax[0].set_ylabel("Magnetization")
ax[0].grid()
ax[1].plot(energy)
ax[1].set_ylabel("Energy")
ax[1].set_xlabel("Number of flipping attempts")
ax[1].grid()
plt.tight_layout()
plt.show()


# In[ ]:




