# -*- coding: utf-8 -*-
"""
Spencer Y. Ki
2023-01-20

Rough work for STA410 HW2
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Problem 1 Functions
def sorted_sum(x, return_cumulative_sequence=False):
    """
    x       : (list) [x[0], x[1], ..., x[n-1]] of numbers
    return_cumulative_sequence : (bool) causes the function to return the 
    cumulative sumation rather than just the final value when set to True
    
    Sums the elements of x in a simple linear manner. To reduce roundoff error 
    x is assumed to be sorted from smallest to largest in absolute value   
    
    return (float) sum of x[0], x[1], ..., x[n-1]
    """
    if return_cumulative_sequence == True:
        holder = []
        s = 0
        for i in x:
            s += i
            holder.append(s)
        return holder
    else:
        s = 0
        for i in x:
            s += i
        return s

def fan(x, minimum_list_length_for_recursion=2):
    """
    x       : (list) [x[0], x[1], ..., x[n-1]] of numbers
    minimum_list_length_for_recursion : (int) the stopping criteria for recursion
    as the function will stop recurring when the list x has shrunk to this value
    
    Sums the elements of x using the recursive fan method. To reduce roundoff 
    error x is assumed to be sorted from smallest to largest in absolute value   
    
    return (float) sum of x[0], x[1], ..., x[n-1]
    """
    if len(x)<minimum_list_length_for_recursion: # stopping criterion
        return sum(x) # `sum(x)` may be used here as provided but not in other functions
    new_x = [] # the new x value to be passed to the next iteration of the fan recursion
    for j in range(int(len(x)/2)): # the loop only needs to cover half the length of x as two elements of x are added for each iteration of the loop
        new_x += [x[j*2]+x[j*2+1]] # during each iteration of the loop 2 elements of x are summed with the result being appended to new_x
    return fan(new_x, minimum_list_length_for_recursion) # new_x is half the length of x and is passed recursively back to restart the function


def kahan(x, return_cumulative_sequence=False):
    """
    x       : (list) [x[0], x[1], ..., x[n-1]] of numbers
    return_cumulative_sequence : (bool) causes the function to return the 
    cumulative sumation rather than just the final value when set to True
    
    Sums the elements of x according to Kahan's improved summation algorithm. 
    To reduce roundoff error x is assumed to be sorted from smallest to largest 
    in absolute value   
    
    return (float) sum of x[0], x[1], ..., x[n-1]
    """
    if return_cumulative_sequence == True:
        holder = [x[0]]
        s = x[0]
        a = 0
        for i in x[1:]:
            y = i - a
            t = s + y
            a = (t - s) - y
            s = t
            holder.append(s)
        return holder
    else:
        s = x[0]
        a = 0
        for i in x[1:]:
            y = i - a
            t = s + y
            a = (t-s)-y
            s = t
        return s

def summation(x, method="kahan", *args, **kwargs):
    """
    x                : (list) [x[0], x[1], ..., x[n-1]] of numbers
    method(="kahan") : (str)  <'sorted'|'fan'|'kahan'> 

    To reduce roundoff error  
    - x is assumed to be sorted from smallest to largest in absolute value
    - 'sorted' sums the elements of x in a simple linear manner
    - 'fan' sums the elements of x pairwise in a recursive manner
    - 'kahan' sums the elements of x according to Kahan's improved summation algorithm.

    return (float) sum of x[0], x[1], ..., x[n-1] using the indicated method
    """
    if method=='sorted':
        if 'return_cumulative_sequence' in kwargs:
            arg_to_forward = kwargs['return_cumulative_sequence']
            return sorted_sum(x, return_cumulative_sequence=arg_to_forward)
        return sorted_sum(x)    
    elif method=='fan':
        if 'minimum_list_length_for_recursion' in kwargs:
            arg_to_forward = kwargs['minimum_list_length_for_recursion']
            return fan(x, minimum_list_length_for_recursion=arg_to_forward)
        return fan(x)
    else:
        if 'return_cumulative_sequence' in kwargs:
            arg_to_forward = kwargs['return_cumulative_sequence']
            return kahan(x, return_cumulative_sequence=arg_to_forward)
        return kahan(x)  

np.random.seed(100)
n = 2**12
x = list(stats.norm().rvs(size=n))
#x = sorted(x, key = lambda x: abs(x))

try:

    fig,ax = plt.subplots(1,2, figsize=(20,5))

    ax[0].plot(np.array(summation(x, method="sorted", return_cumulative_sequence=True))-np.array([sum(x[:(i+1)]) for i in range(len(x))]),
               label="sum")
    ax[0].plot(np.array(summation(x, method="sorted", return_cumulative_sequence=True))-np.array([np.sum(x[:(i+1)]) for i in range(len(x))]),
               label="np.sum")
    ax[0].plot(np.array(summation(x, method="sorted", return_cumulative_sequence=True))-np.array([math.fsum(x[:(i+1)]) for i in range(len(x))]),
               label="math.fsum")
    ax[0].set_title("Difference between method='sorted' and `sum`, `np.sum`, `math.fsum`\n")
    ax[0].legend()

    ax[1].plot(np.array(summation(x, method="kahan", return_cumulative_sequence=True))-np.array([sum(x[:(i+1)]) for i in range(len(x))]),
               label="sum")
    ax[1].plot(np.array(summation(x, method="kahan", return_cumulative_sequence=True))-np.array([np.sum(x[:(i+1)]) for i in range(len(x))]),
               label="np.sum")
    ax[1].plot(np.array(summation(x, method="kahan", return_cumulative_sequence=True))-np.array([math.fsum(x[:(i+1)]) for i in range(len(x))]),
               label="math.fsum")
    ax[1].set_title("Difference between method='kahan' and `sum`, `np.sum`, `math.fsum`\n")
    ax[1].legend()
    
except:
    pass

p1q9_sorted_sum = "sum"
p1q9_kahan = "math.fsum"

# Problem 2 Function
def online(x):
    """
    x                : (list) [x[0], x[1], ..., x[n-1]] of numbers
    
    Calculates the sum of squared errors of x via the online two-pass method. 
    First calculates the mean of x and employs it to calculate the SSE.
    
    return (float) sum of squared errors of x[0], x[1], ..., x[n-1]
    """
    mean = 0
    for i in x:
        mean += i/len(x)
    result = 0
    for i in x:
        result += (i - mean)**2
    return result

def realtime(x):
    """
    x                : (list) [x[0], x[1], ..., x[n-1]] of numbers
    
    Calculates the sum of squared errors of x via the realtime single-pass method. 
    Concurrently sums the squares of x and calculates its mean in a single loop.
    
    return (float) sum of squared errors of x[0], x[1], ..., x[n-1]
    """
    squares = 0
    total = 0
    for i in x:
        squares += i**2
        total += i
    return squares - total/len(x)

def recursive(x):
    """
    x                : (list) [x[0], x[1], ..., x[n-1]] of numbers
    
    Calculates the sum of squared errors of x via an alternative algorithm 
    designed to reduce accumulated roundoff error.
    
    return (float) sum of squared errors of x[0], x[1], ..., x[n-1]
    """
    a = x[0]
    b = 0
    for i in range(2, len(x)+1):
        d = (x[i-1] - a) / i
        a = d + a
        b = i * (i - 1) * d ** 2 + b
    return b

def SSE(x, formula):
    """
    x       : (list) [x[0], x[1], ..., x[n-1]] of numbers
    formula : (str) <'online'|'realtime'|'recursive'> 
    
    To reduce roundoff error  
    - x is assumed to be sorted from smallest to largest in absolute value
    - 'online' calcaulates `xbar` as `sum_i (x[i]/n)` rather than `(sum_i x[i])/n` 
       and `sum_i (x[i]-(xbar))**2` rather than `sum_i x[i]**2 - n*xbar**2`
    - 'realtime' calculates `n*(sum_i (x[i]/n))**2` rather than `(1/n)*(sum_i x[i])**2`
    - 'recursive' as specified is specifically designed to reduce roundoff error.    
    
    return (float) sum of squared errors of x[0], x[1], ..., x[n-1] using the indicated formula
    """
    if formula == 'online':
        return online(x)    
    elif formula == 'realtime':
        return realtime(x)
    else:
        return recursive(x)
    
p2q1 = "D"
p2q2 = "D"

n,mu,std,count1,count2,total_error1,total_error2,trials = 1000,1e1,1e1,0,0,0,0,1000
n,mu,std,count1,count2,total_error1,total_error2,trials = 1000,1e10,1e1,0,0,0,0,1000
for i in range(trials):
    x = stats.norm(loc=mu, scale=std).rvs(size=n)
    x = np.array(sorted(x, key = lambda x: np.abs(x)))
    divide_before = abs((x**2).sum() - n*((x/n).sum()**2) - n*np.var(x, ddof=0))
    divide_after = abs((x**2).sum() - x.sum()**2/n - n*np.var(x, ddof=0))
    if divide_before<divide_after:
        count1 += 1
        total_error1 += divide_before
    if divide_before>divide_after:
        count2 += 1 
        total_error2 += divide_after

finding = "The proportion of trials where `(x**2).sum() - n*((x/n).sum()**2)`\n"
finding += "is more accurate than `(x**2).sum() - x.sum()**2/n` is "+str(count1/trials)+"\n"
finding += "with an average error of "+str(total_error1/count1)+"\n"
finding += "The proportion of trials where `(x**2).sum() - n*((x/n).sum()**2)`\n"
finding += "is less accurate than `(x**2).sum() - x.sum()**2` is "+str(count2/trials)+"\n"
finding += "with an average error of "+str(total_error2/count2)
print(finding)

p2q3 = "D"