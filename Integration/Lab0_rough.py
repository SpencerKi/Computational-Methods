# -*- coding: utf-8 -*-
"""
2023-01-17

PHY408 Lab 0 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

"""
2 Integration Function
"""
# integral function given in lab manual
def integral(y, dx):
    # function c = integral(y, dx)
    # To numerically calculate integral of vector y with interval dx:
    # c = integral[ y(x) dx]
    # ------ This is a demonstration program ------
    n = len(y) # Get the length of vector y
    nx = len(dx) if np.iterable(dx) else 1
    c = 0 # initialize c because we are going to use it
    # dx is a scalar <=> x is equally spaced
    if nx == 1: # ’==’, equal to, as a condition
        for k in range(1, n):
            c = c + (y[k] + y[k-1]) * dx / 2
    # x is not equally spaced, then length of dx has to be n-1
    elif nx == n-1:
        for k in range(1, n):
            c = c + (y[k] + y[k-1]) * dx[k-1] / 2
    # If nx is not 1 or n-1, display an error messege and terminate program
    else:
        print('Lengths of y and dx do not match!')
    return c

# Part 1
# number of samples
nt = 10
# generate time vector
t = np.linspace(0, np.pi/2, nt)
# compute sample interval (evenly sampled, only one number)
dt = t[1] - t[0]
y = np.cos(t)
# plot and save figure
plt.figure(0)
plt.plot(t, y, 'r+')
plt.savefig("Q2P1.pdf")
# compute and print integral
c = integral(y, dt)
print("Calculating the integral of y(t) from 0 to pi/2 yields: " + str(c))

# Part 2
# nt values to loop around given in lab manual
nts = np.array([20, 50, 100, 500, 1000, 5000])
# array to hold values of c for each nt value
cs = np.array([])
# integral for loop
for nt in nts:
    # sampling between [0,0.5]
    t1 = np.linspace(0, 0.5, nt)
    # double sampling between [0.5,1]
    t2 = np.linspace(0.5, 1, 2*nt)
    # concatenate time vector
    t = np.concatenate((t1[:-1], t2))
    # compute y values (f=2t)
    y = np.sin(2 * np.pi * (t + 3 * t**2))
    # plot of y(t) for n(t) = 50
    if nt == 50:
        plt.figure(1)
        plt.plot(t, y)
        plt.savefig("Q2P2.pdf")
    # compute sampling interval vector
    dt = t[1:] - t[:-1]
    c = integral(y, dt)
    cs = np.append(cs, c)
# print array of c values
print("As the value of nt increases, the following trend in c is observed:")
print(cs)
# plot c(nt)
plt.figure(2)
plt.plot(nts, cs)
plt.xlabel("nt")
plt.ylabel("c")
plt.savefig("Q2P2b.pdf")

"""
3 Accuracy of Sampling
"""
# plots that are not to be submitted are commented out

## f = 0
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 0 * t)
plt.figure(3)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 0 * t)
plt.figure(3)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=0")

# f = 0.25
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 0.25 * t)
plt.figure(4)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 0.25 * t)
plt.figure(4)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=0.25")
plt.savefig("Q3P1.pdf")

## f = 0.5
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 0.5 * t)
plt.figure(5)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 0.5 * t)
plt.figure(5)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=0.5")

# f = 0.75
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 0.75 * t)
plt.figure(6)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 0.75 * t)
plt.figure(6)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=0.75")
plt.savefig("Q3P2.pdf")

# f = 1.0
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 1 * t)
plt.figure(7)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 1 * t)
plt.figure(7)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=1.0")

# f = 1.25
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 1.25 * t)
plt.figure(8)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 1.25 * t)
plt.figure(8)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=1.25")

# f = 1.50
dt = 1
t = np.arange(0, 4*np.pi, dt)
g = np.cos(2 * np.pi * 1.50 * t)
plt.figure(9)
plt.plot(t, g, 'rx')

nt = 0.00001
t = np.arange(0, 4*np.pi, nt)
g = np.cos(2 * np.pi * 1.50 * t)
plt.figure(9)
plt.plot(t, g, 'b:')

plt.xlabel("time")
plt.ylabel("g(t)")
plt.title("Sampled Time Series, f=1.50")