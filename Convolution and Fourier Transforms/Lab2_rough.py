# -*- coding: utf-8 -*-
"""
2023-03-06

PHY408 Lab 2 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

#Q1
dt = 1e-3
th_1 = 5
th_2 = 15

#1
def gauss(t, th):
    return np.exp(-(t/th)**2)/(np.sqrt(np.pi)*th)

domain = np.arange(-50, 50, dt)

plt.figure()
plt.plot(domain, gauss(domain, th_1), label = "t_h = 5")
plt.plot(domain, gauss(domain, th_2), label = "t_h = 15")
plt.legend()

#2
def fourier_gauss(w, th):
    return np.exp(-(w**2)*(th**2)/4)

np_trans_1 = np.fft.fft(gauss(domain, th_1)) * dt
np_trans_1 = np.fft.fftshift(np_trans_1)
np_trans_1 = np.abs(np_trans_1)

np_trans_2 = np.fft.fft(gauss(domain, th_2)) * dt
np_trans_2 = np.fft.fftshift(np_trans_2)
np_trans_2 = np.abs(np_trans_2)

f_axis = np.fft.fftshift(np.fft.fftfreq(len(gauss(domain, th_1)), dt))
w_axis = 2*np.pi*f_axis

an_trans_1 = fourier_gauss(w_axis, th_1)
an_trans_2 = fourier_gauss(w_axis, th_2)

plt.figure()
plt.plot(w_axis, an_trans_1, label = "Analytical DFT (t_h = 5)")
plt.plot(f_axis, np_trans_1, label = "Numpy DFT (t_h = 5)")
plt.plot(w_axis, an_trans_2, label = "Analytical DFT (t_h = 15)")
plt.plot(f_axis, np_trans_2, label = "Numpy DFT (t_h = 15)")
plt.legend()

plt.figure()
plt.plot(w_axis[49975:50025], an_trans_1[49975:50025], label = "Analytical DFT (t_h = 5)")
plt.plot(f_axis[49975:50025], np_trans_1[49975:50025], label = "Numpy DFT (t_h = 5)")
plt.plot(w_axis[49975:50025], an_trans_2[49975:50025], label = "Analytical DFT (t_h = 15)")
plt.plot(f_axis[49975:50025], np_trans_2[49975:50025], label = "Numpy DFT (t_h = 15)")
plt.legend()

#Q2
dt = 1e-2
domain = np.arange(-1, 11, dt)
boxcar = np.piecewise(domain, [(domain >= 0) & (domain <= 10)], [1, 0])
hann = np.piecewise(domain, [(domain >= 0) & (domain <= 10)],
                    [lambda domain: 0.5*(1 - np.cos(2*np.pi*domain/10)), 0])

#1
plt.figure()
plt.plot(domain, boxcar, label = "Boxcar")
plt.plot(domain, hann, label = "Hann")
plt.legend()

#2
trans_box = np.fft.fft(boxcar) * dt
trans_box = np.fft.fftshift(trans_box)
trans_box = np.abs(trans_box)

trans_hann = np.fft.fft(hann) * dt
trans_hann = np.fft.fftshift(trans_hann)
trans_hann = np.abs(trans_hann)

f_axis = np.fft.fftshift(np.fft.fftfreq(len(hann), dt))

#3
plt.figure()
plt.plot(f_axis, trans_box, label = "Boxcar")
plt.plot(f_axis, trans_hann, label = "Hann")
plt.legend()

plt.figure()
plt.plot(f_axis[575:625], trans_box[575:625], label = "Boxcar")
plt.plot(f_axis[575:625], trans_hann[575:625], label = "Hann")
plt.legend()

#Q4
