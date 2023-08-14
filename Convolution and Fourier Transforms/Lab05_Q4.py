# -*- coding: utf-8 -*-
"""
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt

# Import photo and save side length
blur = np.loadtxt('blur.txt')
side = 1024

# Plot blurred photo
plt.figure(0)
plt.imshow(blur, cmap='gray')

# Gaussian point-spread function given in text
def gaussian(x, y, sigma):
    return np.exp(-(x**2+y**2)/(2*sigma**2))

# Accounting for periodic behaviour of point-spread
def periodic(z):
    return (z + side/2)%side - side/2

# Generating point-spread array
point_spread = np.zeros(np.shape(blur))
for i in range(side):
    for j in range(side):
        point_spread[i][j] = gaussian(periodic(i), periodic(j), 25)
        
# Plot point-spread
plt.figure(1)
plt.imshow(point_spread, cmap='gray')
        
# RFFT of point-spread array and blurred photo
b = np.fft.rfft2(blur)
f = np.fft.rfft2(point_spread)

# Applying deconvolution step accounting for precision error
holder = np.zeros([side, side//2 + 1], dtype = complex)
for i in range(side):
    for j in range(side//2 + 1):
        if np.abs(f[i][j]) > 10e-3:
            holder[i][j] = b[i][j]/(f[i][j])
        else:
            holder[i][j] = b[i][j]

# Applying IRFFT and plot unblirred photo
unblur = np.fft.irfft2(holder)
plt.figure(2)
plt.imshow(unblur, cmap='gray')