# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 07:29:22 2021

@author: spenc
"""

from math import sqrt,log,cos,sin,pi
from random import random
# Constants
Z = 104
e = 1.602e-19
E = 7.7e6*e
epsilonO = 8.854e-12
a0 = 5.292e-11
sigma = a0/100
N = 1000000
# Function to generate two Gaussian random numbers
def gaussian():
    r = sqrt(-2*sigma*sigma*log(1-random()))
    theta = 2*pi*random()
    x = r*cos(theta)
    y = r*sin(theta)
    return x,y
# Main program
count = 0
for i in range(N):
    x,y = gaussian()
    b = sqrt(x*x+y*y)
    if b<Z*e*e/(2*pi*epsilonO*E):
            count += 1
print(count,"particles were reflected out of",N)