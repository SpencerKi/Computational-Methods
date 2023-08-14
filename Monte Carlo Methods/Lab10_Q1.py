#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Question 1 d)
# Author: Ilyas Sharif

# Importing required packages
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from random import random
from matplotlib import cm

# Loading in the land data
loaded = np.load('Earth.npz')
data = loaded['data']
lon_array = loaded['lon']
lat_array = loaded['lat']

# Converting to Radians
lon_array = ((lon_array+180)/180)*np.pi
lat_array = ((lat_array+90)/180)*np.pi

# Defining our functions that calulcate the angles theta, phi
# given some random number z
def ftheta(z):
    return np.arccos(1 - 2*z)
def fphi(z):
    return 2*np.pi*z

# Number of sample points and empty arrays for 2D plot
N = 50000 ### Note: to get the different plots, just change this value ###
land_points = []
water_points = []

# Create nearest interpolator
interp = RegularGridInterpolator((lon_array, lat_array), data, method='nearest')

# Creating points on the "globe"
for i in range(N):
    # Create two random numbers to feed into our theta and phi functions
    theta = ftheta(random())
    phi = fphi(random())
    
    # creating a fix for values above the maximum longtitude
    if phi > lon_array[len(lon_array)-1]:
        phi = lon_array[len(lon_array)-1]
    
    # Computing the interpolation of the points
    # determines if they are near land or not
    delta = interp([phi, theta])
    if delta == 1:
        land_points.append([phi, theta])

    else:
        water_points.append([phi, theta])


# Chaning to numpy arrays to get more functionality
land_points = np.array(land_points)
water_points = np.array(water_points)

# Printing the land fraction
print("The land fraction is = " + str(len(land_points)/(len(land_points) + len(water_points))) + " for N = " + str(N))

# Creating the 2D plots
plt.scatter(land_points[:,0], land_points[:,1], color = 'g', marker = '.')
plt.scatter(water_points[:,0], water_points[:,1], color = 'b' , marker = '.')
plt.xlabel("Longitude (Radians)")
plt.ylabel("Latitude (Radians)")
plt.title("Random Location Generator - Land / Water points")
plt.show()


# In[ ]:




