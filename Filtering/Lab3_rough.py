# -*- coding: utf-8 -*-
"""
2023-04-26

PHY408 Lab 3 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

#Q1
# Parameters
fs = 12.0
f0 = 1.0
M = 1.05
e = 0.05

# Zeros and poles
q = np.exp(-2j * np.pi * f0 / fs)
p = (1 + e) * q

# Compute frequency response
w = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
z = np.exp(1j * w)
Wz = M * ((z - q) / (z - p)) * ((z - np.conj(q)) / (z - np.conj(p)))
P = np.abs(Wz)**2

# Plot power spectrum
fig, ax = plt.subplots()
ax.plot(w / (2 * np.pi) * fs, P)
ax.set_xlabel('Frequency (cycles/year)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum of Notch Filter')
plt.show()

#Q2
def ratFilter(N, D, x):
    y1 = np.convolve(N, x, mode='full')
    y2 = np.convolve(D, y1, mode='full')
    return y2[:len(x)]

# Compute the impulse response of the filter
N = [1.05, -1.8186533479473213, 1.05]
D = [1, -1.8186533479473213, 1]
t = np.arange(0, 100, 1/12)
impulse = np.zeros(1200)
impulse[0] = 1
impulse_response = ratFilter(N, D, impulse)

# Compute the frequency response of the filter
H = np.fft.fft(impulse_response)

# Compute the theoretical spectrum
f = np.linspace(0, 6, 1000)
w = 2 * np.pi * f
z = np.exp(-1j * w)
P = (1.05 - 1.8186533479473213 * z + 1.05 * z**2) / (1 -1.8186533479473213 + z**2)

# Plot the frequency response and theoretical spectrum
plt.figure()
plt.plot(f, np.abs(H)[:len(f)])
plt.plot(w, np.abs(P))
plt.xlabel('Frequency (cycles/year)')
plt.ylabel('Magnitude')
plt.legend(['Computed', 'Theoretical'])
plt.title('Frequency Response of Notch Filter')
plt.show()

#Q3
import co2data as co2
dat = co2.co2Data
time = np.linspace(co2.co2TimeRange[0], co2.co2TimeRange[1], len(co2.co2Data))

straight = np.polyfit(time, dat, 1)
detrend = dat - np.polyval(straight, time)

plt.figure()
plt.plot(time, dat, label = "Original")
plt.plot(time, detrend, label = "Detrended")
plt.xlabel("Time (months)")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("CO2 Concentration, Original vs. Detrended Data")
plt.legend()

filtered = ratFilter(N, D, detrend)
retrend = filtered + np.polyval(straight, time)

ft_detrend = np.fft.fft(detrend)
f = np.fft.fftfreq(len(detrend), 1/12)

amp_ft_detrend = np.abs(ft_detrend)
phase_ft_detrend = np.angle(ft_detrend)

ft_detrend_9 = ft_detrend
ft_detrend_9[(f > 0.9) | (f < -0.9)] = 0
detrend_9 = np.real(np.fft.ifft(ft_detrend_9))