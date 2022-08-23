import scipy.optimize as sp
import numpy as np
from scipy.special import digamma, polygamma
import matplotlib.pyplot as plt

t = np.array([8.22, 16.26, 13.09, 8.33, 9.62, 10.10, 7.86, 5.41, 8.28, 9.98, 4.47, 9.89, 8.16, 5.21, 3.82, 18.71, 9.48, 4.89, 11.74, 6.28])

s = np.log(np.sum(t)/20) - np.sum(np.log(t))/20

def optimizable(k):
    return digamma(k) - np.log(k) + s

def derivada(k):
    return polygamma(1, k) - 1/k

initial = (3 - s + np.sqrt((s-3)**2 + 24*s))/(12*s) # aproximación de la raíz (wikipedia)

k = sp.root_scalar(optimizable, x0=initial, fprime=derivada, method='newton')
k = k.root
b = 20*k/np.sum(t)
print(k, b)

plt.figure()
plt.plot(np.linspace(5, 9, 100), optimizable(np.linspace(5,9,100)))
plt.show()