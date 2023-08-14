# -*- coding: utf-8 -*-
"""
2023-01-14

STA410 HW1 Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def binary_string(integer_input, storage_bits=16):
    """    
    integer_input     : (int) must be an integer which may be negative
    storage_bits(=16) : (int) representational storage capacity in bits

    returns (str) signed (`storage_bits`-)bit representation of `integer_input`
    """
    if type(integer_input) != int:
        return "Error: this function takes integers only."

    maximum_representable_integer = 2**(storage_bits - 1) - 1

    if abs(integer_input) > maximum_representable_integer:
        return "Error: `integer_input` exceeds the representational capability of `storage_bits`."

    bits = storage_bits*[0]
    
    if integer_input < 0:
        bits[0] = 1
        integer_input = abs(integer_input)
        
    remainder = integer_input
    for i in range(1, storage_bits):
        bits[i - storage_bits] = remainder // 2**(storage_bits - i - 1)
        remainder = remainder % 2**(storage_bits - i - 1)

    return "".join([str(bit) for bit in bits])

p1q0 = binary_string(-7, 16)[0]+binary_string(7, 16)[0]+binary_string(-2**12, 16)[0]+binary_string(-2**11, 16)[0]
p1q1 = binary_string(21845, 16)[10:14]
p1q2 = binary_string(-10922, 16)[-11:-5]
p1q3 = binary_string(32768, 16)[-6]

# Problem 2
def create_integer_sampler(a, m):
    """
    a : (int) multiplier
    m : (int) modulus
    
    This function creates and returns an iteratively-implemented function for pseudorandom number generation
    using the first order sequential congruential method u_k = au_{k-1} mod m, 0<|u_k|<m 
    """
    def pseudorandom_integer_sample(n, pseudorandom_sample):
        """
        n : (int) desired sample size
        pseudorandom_sample : (list) list initialised with a single int 'seed value.'
        This seed value is the start of the iterative pseudorandom number generation process.
        
        This is an iterative function for pseudorandom number generation using the 
        first order sequential congruential method u_k = au_{k-1} mod m, 0<|u_k|<m 
        a and m are inherited from the parent function.
        """
        for i in range(n - 1):
            next_pseudorandom_sample = a * pseudorandom_sample[-1] % m
            if next_pseudorandom_sample in pseudorandom_sample:
                print("Warning: sampling period reached -- psuedorandom samples are being repeated")
                return pseudorandom_sample
            pseudorandom_sample += [next_pseudorandom_sample]
        return pseudorandom_sample
    return pseudorandom_integer_sample

p2q1a = "No"
p2q1b = "B"

my_psuedor_sampler = create_integer_sampler(a=5, m=7921)
seed = 1234567890
num = 1000000
p2q2 = len(my_psuedor_sampler(n=num, pseudorandom_sample=[seed])) 
p2q3 = len(create_integer_sampler(a=811, m=7921)(n=num, pseudorandom_sample=[seed]))
p2q4 = len(create_integer_sampler(a=814, m=7921)(n=num, pseudorandom_sample=[seed]))
p2q5 = len(create_integer_sampler(a=826, m=7921)(n=num, pseudorandom_sample=[seed]))

try: 
    def label_plot(ax):
        ax.set_xlabel("x[i]"); ax.set_ylabel("x[i+1]")
        
    fig,ax = plt.subplots(1,4,figsize=(20,5))
    x = np.array(create_integer_sampler(a=5, m=7921)(n=8000, pseudorandom_sample=[1]))
    ax[0].plot(x[:100],x[1:101],'.'); ax[0].set_title("a=5 m=7921 period="+str(len(x))); label_plot(ax[0])
    x = np.array(create_integer_sampler(a=811, m=7921)(n=8000, pseudorandom_sample=[1100]))
    ax[1].plot(x[:100],x[1:101],'.'); ax[1].set_title("a=811 m=7921 period="+str(len(x))); label_plot(ax[1])
    x = np.array(create_integer_sampler(a=814, m=7921)(n=8000, pseudorandom_sample=[1100]))
    ax[2].plot(x[:100],x[1:101],'.'); ax[2].set_title("a=814 m=7921 period="+str(len(x))); label_plot(ax[2])
    x = np.array(create_integer_sampler(a=826, m=7921)(n=8000, pseudorandom_sample=[1100]))
    ax[3].plot(x[:-1],x[1:],'.'); ax[3].set_title("a=826 m=7921 period="+str(len(x))); label_plot(ax[3])
except:
    pass
p2q6 = "C"

a = len(create_integer_sampler(a=7907, m=7919)(n=num, pseudorandom_sample=[seed]))
b = len(create_integer_sampler(a=3967, m=7921)(n=num, pseudorandom_sample=[seed]))
c = len(create_integer_sampler(a=814, m=7883)(n=num, pseudorandom_sample=[seed]))
p2q7 = "A"

p2q8 = "No"
p2q9 = "No"