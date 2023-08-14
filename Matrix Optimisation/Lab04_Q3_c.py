# -*- coding: utf-8 -*-
"""
Authors: Spencer Ki, Ilyas Sharif
"""
import numpy as np
import matplotlib.pyplot as plt

# Equation from text
def displace(x):
    return 5*np.exp(-x) + x - 5

# Modified equation from text
def displace2(x):
    return 5 - 5*np.exp(-x)

# Derivative of above for Newton
def ddx_displace(x):
    return 5*np.exp(-x)

# Binary search
def binary(init_1, init_2):
    xp = init_1
    xn = init_2
    mid = displace(0.5*(xp+xn))
    counter = 0
    if mid > 0:
        xp = 0.5*(xp+xn)
    else:
        xn = 0.5*(xp+xn)
    
    while np.abs(mid) > 1e-6:
        mid = displace(0.5*(xp+xn))
        if mid > 0:
            xp = 0.5*(xp+xn)
        else:
            xn = 0.5*(xp+xn)
        counter += 1
        
    return mid, counter

# Relaxation method
def relax(guess):
    est = displace2(guess)
    counter = 0
    while np.abs(displace2(est) - est) > 1e-6:
        est = displace2(est)
        counter += 1
    return est, counter

# Newton method
def Newton(guess):
    est = guess - displace2(guess)/ddx_displace(guess)
    counter = 0
    while np.abs(displace2(est) - est) > 1e-6:
        est = est - displace2(est)/ddx_displace(est)
        counter += 1
    return est, counter