from math import sqrt,exp
from numpy import empty
from random import random,randrange,seed
import matplotlib.pyplot as plt

seed(112358)

N = 25
R = 0.02
Tmax = 10.0
Tmin = 1e-3
tau = 1e5

# Function to calculate the magnitude of a vector
def mag(x):
    return sqrt(x[0]**2+x[1]**2)

# Function to calculate the total length of the tour
def distance():
    s = 0.0
    for i in range(N):
        s += mag(r[i+1]-r[i])
    return s

# Choose N city locations and calculate the initial distance
r = empty([N+1,2],float)
for i in range(N):
    r[i,0] = random()
    r[i,1] = random()
r[N] = r[0]
D = distance()

# Main loop
fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040]
for k in fib:
    seed(k)
    
    t = 0
    T = Tmax
    while T>Tmin:
    
        # Cooling
        t += 1
        T = Tmax*exp(-t/tau)
    
        # Choose two cities to swap and make sure they are distinct
        i,j = randrange(1,N),randrange(1,N)
        while i==j:
            i,j = randrange(1,N),randrange(1,N)
    
        # Swap them and calculate the change in distance
        oldD = D
        r[i,0],r[j,0] = r[j,0],r[i,0]
        r[i,1],r[j,1] = r[j,1],r[i,1]
        D = distance()
        deltaD = D - oldD
    
        # If the move is rejected, swap them back again
        if random()>exp(-deltaD/T):
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            D = oldD
    
#    plt.figure()
#    plt.scatter(*zip(*r))
#    plt.plot(*zip(*r))
#    plt.title('While Loop Seed = {0:.1f}'.format(k))
    
    print(D)