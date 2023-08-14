# Authors: Spencer Ki, Ilyas Sharif

#Question 1b
import numpy as np

# Loading the provided data and calculating its true standard deviation.
cdata = np.loadtxt('cdata.txt')
true_std = np.std(cdata, ddof=1)

# Formula 1 for calculating standard deviation, as described in pseudocode.
def formula1(data):
    mean = np.sum(data) / len(data)
    placeholder = 0
    for datum in data:
        placeholder += (datum - mean)**2
    return np.sqrt(placeholder/(len(data) - 1))

# Formula 2 for calculating standard deviation, as described in pseudocode
def formula2(data):
    s = 0
    m = 0
    for datum in data:
        s += np.round(datum**2,4)
        m += datum
    placeholder = s - np.round(m**2,4)/len(data)
    if placeholder < 0:
        return "ERROR"
    else:
        return np.sqrt(placeholder/(len(data) - 1))

# Print output. Home-script check since functions are called in other scripts.
if __name__ == "__main__":
    err1 = (formula1(cdata) - true_std) / true_std
    err2 = (formula2(cdata) - true_std) / true_std
    print("Formula 1 yields a relative error of " + str(err1))
    print("Formula 2 yields a relative error of " + str(err2))