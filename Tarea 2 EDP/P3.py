import numpy as np
import matplotlib.pyplot as plt
N = 10
L = 2*np.pi
dx = L/N

def v(x, i):
    xant = (i - 1)*dx
    xi = i*dx
    xsig = (i + 1)*dx
    returnable = np.zeros(len(x))
    returnable[np.logical_and(x > xant,x < xi)] = (x[np.logical_and(x > xant, x < xi)] - xant)/dx
    returnable[np.logical_and(x >= xi, x < xsig)] = (xsig - x[np.logical_and(x >= xi, x < xsig)])/dx
    return returnable

x = np.linspace(0, L, 100)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(N):
    ax.plot(x, v(x, i))
    
