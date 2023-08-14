# Authors: Ilyas Sharif, Spencer Ki

#Question 3
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Manual histogram creation
def histo_maker(inputs):
    bin_edges = np.linspace(-5, 5, 1001)# Define bin edges
    bins = np.zeros(1000)# The empty bins
    
    for num in inputs:
        # For each number, the loop starts in the middle bin and iterates
        # upwards or downwards until the appropriate bin is found.
        binned = False
        index = 500
        while not binned:
            if num < bin_edges[index]:
                index -= 1
            elif num >= bin_edges[index + 1]:
                index += 1
            else:
                bins[index] += 1
                binned = True

    return bins

counts_list = np.array([10, 100, 1000, 10000, 100000, 1000000])# N counts
manual_times = np.empty(0)# place-holder array for manual generation times
nphist_times = np.empty(0)# place-holder array for Numpy generation times

for counts in counts_list:
    rand_nums = np.random.randn(counts)
    
    # Since numbers outside of (-5, 5) could be generated randomly, this
    # loop just drops those values from the list if they come up
    for i in range(len(rand_nums)):
        if rand_nums[i] < -5 or rand_nums[i] >= 5:
            rand_nums = np.delete(rand_nums, i)
    
    # Time the manual histogram creation method
    start_time = time()
    manual = histo_maker(rand_nums)
    end_time = time()
    manual_times = np.append(manual_times, end_time - start_time)
    
    # Time the Numpy histogram creation method
    start_time = time()
    nphist = np.histogram(rand_nums, 1000, (-5, 5))[0]
    end_time = time()
    nphist_times = np.append(nphist_times, end_time - start_time)
    
    # Confirm that the manual and Numpy generated histograms are equivalent
    if not np.array_equal(manual, nphist):
        print("HISTOGRAMS ARE NOT EQUIVALENT.")
        break

# Plot results
plt.figure(0)
plt.plot(counts_list, manual_times, label = "Manual function")
plt.plot(counts_list, nphist_times, label = "Numpy function")
plt.yscale("log")
plt.legend()
plt.title("Manual vs. Numpy Histogram Creation Time")
plt.xlabel("Number of Random Samples")
plt.ylabel("Time in Log-Scale (s)")
plt.savefig("generation_time.pdf")

# Print manual generation times for Question 3a
print(manual_times)