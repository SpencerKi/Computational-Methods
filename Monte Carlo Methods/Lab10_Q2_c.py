# -*- coding: utf-8 -*-
"""
PHY407 Lab 10, Question 2c
Spencer Ki, Ilyas Sharif
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# %% Functions taken from lab manual -----------------------------------------|
def get_tau_step():
    """ Calculate how far a photon travels before it gets scattered.
    OUT: optical depth traveled """
    delta_tau = -np.log(np.random.random())
    return delta_tau

def emit_photon(tau_max):
    """ Emit a photon from the stellar core.
    IN: tau max is max optical depth
    OUT:
    tau: optical depth at which the photon is created
    mu: directional cosine of the photon emitted """
    tau = tau_max
    delta_tau = get_tau_step()
    mu = np.random.random()
    return tau - delta_tau*mu, mu

def scatter_photon(tau):
    """ Scatter a photon.
    IN: tau, optical depth of the atmosphere
    OUT:
    tau: new optical depth
    mu: directional cosine of the photon scattered """
    delta_tau = get_tau_step()
    mu = 2*np.random.random()-1 # sample mu uniformly from -1 to 1
    return tau - delta_tau*mu, mu

# %% Original code begins here -----------------------------------------------|

# Function to simulate photon emission and scattering
def photon_path(tau_max):
    tau, mu = emit_photon(tau_max)
    count = 0
    while tau > 0:
        count += 1
        tau, mu = scatter_photon(tau)
        if tau > tau_max:
            tau, mu = emit_photon(tau_max)
            count = 0
    return count, mu

# Scenario parameters
tau_max = 1e-4
data_num = 1e5
n_bins = 20
bin_size = int(data_num/n_bins)

# Simulate data_num number of photons and store their mu values
mu_holder = np.empty(0)
for i in range(int(data_num)):
    mu_holder = np.append(mu_holder, photon_path(tau_max)[1])

# Sort mu values into histogram bins
histo = np.histogram(mu_holder, bins = n_bins)

# Calculate centre of each histogram bin
av_mu_holder = (histo[1][1:] + histo[1][:-1])/2

# Calculate specific intensity from histogram volume and central mu values
spec_inten = histo[0]/av_mu_holder

# Normalise specific intensity to I_1
norm_spec = spec_inten / spec_inten[-1]

# Best fit calculation for angular dependency relation
def intensity(mu, a, b):
    return a + b*mu
fitted = opt.curve_fit(intensity, av_mu_holder, norm_spec)

# Plotting
plt.figure(0)
plt.hist(mu_holder, bins = n_bins)
plt.xlabel('Direction Cosine (\u03BC)')
plt.ylabel('Incidences of \u03BC Values')
plt.title('Incidences of \u03BC Values of 100,000 Simulated Photons')

plt.figure(1)
plt.scatter(av_mu_holder, norm_spec, label = 'Bin Midpoint Values')
plt.plot(mu_holder, intensity(mu_holder, 0.4, 0.6), label = 'Analytic Fit')
plt.plot(mu_holder, intensity(mu_holder, fitted[0][0], fitted[0][1]), label = 'Best Fit')
plt.legend()
plt.xlabel('Direction Cosine (\u03BC)')
plt.ylabel('Normalized Specific Intensities (I(\u03BC)/I_1)')
plt.title('Normalised Specific Intensity as a Function of Direction Cosine')