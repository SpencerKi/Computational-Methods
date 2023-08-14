# -*- coding: utf-8 -*-
"""
2023-05-02

PHY408 Lab 4 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

#Q1
# Load data
phl = np.loadtxt('PHL_data.txt')
mlac = np.loadtxt('MLAC_data.txt')

# Cross-correlate
phl_ft = np.fft.fft(phl).flatten()
mlac_ft = np.fft.fft(mlac).flatten()
cross_corr = np.fft.ifft(phl_ft * np.conj(mlac_ft))
t = np.arange(0, 86400, 1)

# # Plot
plt.plot(t, np.real(cross_corr))
plt.xlim([0, 300])
plt.xlabel('Time (s)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between PHL and MLAC')

#Q2
dat = np.loadtxt('nwao.vh1', unpack = True)
t = dat[0] / 3600
v = dat[1]
f = np.fft.fftfreq(len(v), 10) * 1000
spec = np.abs(np.fft.fft(v))**2

# Apply Hanning window
window = 0.5 * (1 - np.cos(2 * np.pi * t) / (len(v)))
v = v * window

# Compute power spectrum
dt = t[1] - t[0]
fft = np.fft.fft(v)
freq = np.fft.fftfreq(len(v), d=dt)
power = np.abs(fft)**2 / (dt * len(v))

# Plot results
plt.figure()
plt.plot(freq * 1000, power)
plt.xlabel('Frequency (mHz)')
plt.ylabel('Power')
plt.show()