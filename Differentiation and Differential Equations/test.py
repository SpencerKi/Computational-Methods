# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 23:35:57 2021

@author: spenc
"""

from numpy import empty,zeros,max,printoptions,inf, linspace
from pylab import imshow,gray,show
# Constants
M = 200
N = 80
a = 0.01
target = 1e-6
phi= zeros([M+1,N+1] ,float)

omega = 0.9
# Main loop
delta = 1.0
philist=[]
while delta>target:
    #print (delta)
    delta = 0.0
    phifinal=[]
    for i in range(1, M): 
        for j in range(1, N): 
            if i == N-1 and j>0 and j<50: 
                phi[N,0:N+1]=linspace(0,5,N+1)
            elif j==50 and i>50 and i<N:
                phi[50:N,50]=linspace(7,5,30)
            elif i==50 and j>50 and j<150:
                phi[50,50:150]= 7
            elif i == 0+1 and j>150 and j<M:
                phi[1,0:M]=10
            elif i == N-1 and j>150 and j<M:
                phi[N,150:M]=linspace(5,0,50)
            elif j==150 and i>50 and i<N:
                phi[50:N,150]=linspace(7,5,30)
            elif j == M-1 and i>50 and i<N:
                phi[0:N,(M-1)]=linspace(10,0,N)
            elif j>50 and j<150 and i>50 and i<N:
                phi[i,j]=0
            elif j ==0+1 and i>0 and i<N:
                phi[0:N,1]=linspace(10,0,N)
            else:
                diff= abs((phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - phi[i,j])
                phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - (omega)*phi[i,j]
                if diff>delta:
                    delta = diff            

    phifinal.append(phi)
#print (phi)            
imshow(phi,cmap='inferno')
show()  