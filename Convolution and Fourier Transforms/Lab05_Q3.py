# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt

SLP = np.loadtxt('SLP.txt')
Longitude = np.loadtxt('lon.txt')
Times = np.loadtxt('times.txt')

plt.figure(0)
plt.contourf(Longitude, Times, SLP)
plt.xlabel('longitude(degrees)')
plt.ylabel('days since Jan. 1 2015')
plt.title('SLP anomaly (hPa)')
plt.colorbar()

# RFFT of SLP
transform = np.fft.rfft(SLP)

# Copies of the transform
dat_3 = np.copy(transform)
dat_5 = np.copy(transform)

# Isolating the wavenumbers desired
for i in range(120):
    for j in range(73):
        if j != 3:
            dat_3[i][j] = 0
        if j != 5:
            dat_5[i][j] = 0
         
# Taking the iRFFTs of the isolated wavenumbers
inverse_3 = np.fft.irfft(dat_3)
inverse_5 = np.fft.irfft(dat_5)

plt.figure(1)
plt.contourf(Longitude, Times, inverse_3)
plt.xlabel('longitude(degrees)')
plt.ylabel('days since Jan. 1 2015')
plt.title('SLP anomaly (hPa), m=3')
plt.colorbar()

plt.figure(2)
plt.contourf(Longitude, Times, inverse_5)
plt.xlabel('longitude(degrees)')
plt.ylabel('days since Jan. 1 2015')
plt.title('SLP anomaly (hPa), m=5')
plt.colorbar()

plt.figure(3)
plt.plot(inverse_3)
plt.xlabel('days since Jan. 1 2015')
plt.ylabel('SLP anomaly (hPa)')
plt.title('SLP anomaly, m=3 vs. Longitude')

plt.figure(4)
plt.plot(inverse_5)
plt.xlabel('days since Jan. 1 2015')
plt.ylabel('SLP anomaly (hPa)')
plt.title('SLP anomaly, m=5 vs. Longitude')

# Speed calculation
circumference = 2*np.pi*6.371e6*np.cos(50*np.pi/180)
seconds = 80*24*3600
speed = circumference/seconds