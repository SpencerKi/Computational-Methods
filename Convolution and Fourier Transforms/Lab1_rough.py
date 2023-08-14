# -*- coding: utf-8 -*-
"""
2023-02-06

PHY408 Lab 1 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt
import time

#Q1
def myConv(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    f: (np.ndarray) A 1-dimensional numeric array of length greater than 0.
    w: (np.ndarray) A 1-dimensional numeric array of length greater than 0.
    return: (np.ndarray) A 1-dimensional numeric array of length len(f) + len(w) - 1
    
    Convolves arrays f and w and returns the convolution.
    """
    # Condition identifies the longer array and appends appropriate zeroes to it.
    if len(f) > len(w):
        shorter = np.flip(w)
        longer = np.append(np.zeros(len(shorter) - 1), 
                           np.append(f, np.zeros(len(shorter) - 1)))
    else:
        shorter = np.flip(f)
        longer = np.append(np.zeros(len(shorter) - 1), 
                           np.append(w, np.zeros(len(shorter) - 1)))
    
    result = []
    # Outer loop iterates through all the values in the convolution array.
    for i in range(len(f) + len(w) - 1):
        start = i
        products = 0
        # Inner loop calculates entries in convolution array via repeated addition.
        for j in range(len(shorter)):
            products += longer[start] * shorter[j]
            start += 1
        result.append(products)
    
    return np.array(result)

# test_f = np.random.rand(75)
# test_w = np.random.rand(150)
# np_test = np.convolve(test_f, test_w)
# my_test = myConv(test_f, test_w)

# sizes  = [10, 100, 1000, 10000]
# times = []
# for size in sizes:
#     test_f = np.random.rand(size)
#     test_w = np.random.rand(size)
    
#     t1 = time.time()
#     np.convolve(test_f, test_w)
#     t2 = time.time()
    
#     t3 = time.time()
#     myConv(test_f, test_w)
#     t4 = time.time()
    
#     times.append(t4-t3+t2-t1)

#Q3
TES_data = np.loadtxt("TES_Spectra.txt", unpack=True)
wavenumbers = TES_data[0]
spectra = TES_data[1]

def blurrer(wavenumber: float, resolution: int) -> float:
    """
    wavenumber: (float) A float representing a wavenumber.
    resolution: (int) An int representing the spectral resolution.
    return: (float) A float representing the 'blurred' spectrum.
    
    'Blurs' the spectrum by the given resolution as described in the lab manual.
    """
    return 2 * resolution * np.sin(2 * np.pi * wavenumber * resolution) / \
        (2 * np.pi * wavenumber * resolution)

interval = np.arange(-3.0, 3.01, 0.06)
res_1 = blurrer(interval, 1)
res_3 = blurrer(interval, 3)

conv_1 = np.convolve(spectra[834:1666], res_1, mode = "same")
conv_3 = np.convolve(spectra[834:1666], res_3, mode = "same")

# plt.figure()
# plt.plot(wavenumbers[834:1666], conv_1)
# plt.plot(wavenumbers[834:1666], spectra[834:1666])
# plt.xlabel("Wavenumber (1/cm)")
# plt.title("Curve Convolution with Spectrum, Wavenumber = [700, 750], Resolution = 1")

# plt.figure()
# plt.plot(wavenumbers[834:1666], conv_3)
# plt.plot(wavenumbers[834:1666], spectra[834:1666])
# plt.xlabel("Wavenumber (1/cm)")
# plt.title("Curve Convolution with Spectrum, Wavenumber = [700, 750], Resolution = 3")

def gaussian(wavenumber: float, L: int) -> float:
    """
    wavenumber: (float) A float representing a wavenumber.
    L: (int) An int parameter for the Gaussian function.
    return: (float) A float representing a 'Gaussian blur' of the spectrum.
    
    'Gaussian blurs' the spectrum by the given parameter as described in the lab manual.
    """
    return np.exp(-(wavenumber / L)**2) / (np.sqrt(np.pi) * L)

#Q2
# H = np.append(0.5, np.ones(len(V_in) - 1))
# D = np.append(1/dt, np.zeros(len(V_in) - 1))
Heav = np.append(0.5, np.ones(99))
Dirac = np.append(1/0.00025, np.zeros(99))


def step(R: int, L: int, V_in: np.ndarray, dt: float) -> np.ndarray:
    """
    R: (int) An int representing the circuit resistor's resistance.
    L: (int) An int representing the circuit inductor's inductance.
    V_in: (np.ndarray) An array representing an input series of voltages.
    dt: (float) A float representing the sampling period.
    returns: (np.ndarray) An array representing the output series of voltages.
    
    Calculates the voltages in the circuit system.
    """
    T = np.arange(0, len(V_in) * dt, dt)# Time values, properly spaced
    H = np.append(0.5, np.ones(len(V_in) - 1))# Discrete Heaviside
    return np.exp(-R * T / L) * H

def impulse(R: int, L: int, V_in: np.ndarray, dt: float) -> np.ndarray:
    """
    R: (int) An int representing the circuit resistor's resistance.
    L: (int) An int representing the circuit inductor's inductance.
    V_in: (np.ndarray) An array representing an input series of voltages.
    dt: (float) A float representing the sampling period.
    returns: (np.ndarray) An array representing the circuit's impulse response.
    
    Calculates the impulse response of the circuit.
    """
    T = np.arange(0, len(V_in) * dt, dt)# Time values, properly spaced
    H = np.append(0.5, np.ones(len(V_in) - 1))# Discrete Heaviside
    D = np.append(1/dt, np.zeros(len(V_in) - 1))# Discrete Dirac delta
    return D - R/L * np.exp(-R * T / L) * H

def RLresponse(R: int, L: int, V_in: np.ndarray, dt: float) -> np.ndarray:
    """
    R: (int) An int representing the circuit resistor's resistance.
    L: (int) An int representing the circuit inductor's inductance.
    V_in: (np.ndarray) An array representing an input series of voltages.
    dt: (float) A float representing the sampling period.
    returns: (np.ndarray) An array representing the output series of voltages.
    
    Calculates the output voltages over the inductor.
    """
    return np.convolve(impulse(R, L, V_in, dt), V_in, mode = "same") * dt

# plt.figure()
# plt.plot(step(1000,3,Heav,0.00025))

# plt.figure()
# plt.plot(RLresponse(1000,3,Heav,0.00025))

# plt.figure()
# plt.plot(impulse(1000,3,Heav,0.00025))