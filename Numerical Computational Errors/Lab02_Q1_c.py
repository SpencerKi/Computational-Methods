# Authors: Spencer Ki, Ilyas Sharif

#Question 1c
import numpy as np
from Lab02_Q1_b import formula1, formula2

set1 = np.random.normal(0., 1., 2000)
set2 = np.random.normal(1.e7, 1., 2000)

set1_err1 = (formula1(set1) - np.std(set1, ddof=1)) / np.std(set1, ddof=1)
set1_err2 = (formula2(set1) - np.std(set1, ddof=1)) / np.std(set1, ddof=1)
set2_err1 = (formula1(set2) - np.std(set2, ddof=1)) / np.std(set2, ddof=1)
set2_err2 = (formula2(set2) - np.std(set2, ddof=1)) / np.std(set2, ddof=1)

print("Formula 1 yields a relative error of " + str(set1_err1) + \
      " for the set with mean 0.")
print("Formula 2 yields a relative error of " + str(set1_err2) + \
      " for the set with mean 0.")
print("Formula 1 yields a relative error of " + str(set2_err1) + \
      " for the set with mean 1.e7.")
print("Formula 2 yields a relative error of " + str(set2_err2) + \
      " for the set with mean 1.e7.")