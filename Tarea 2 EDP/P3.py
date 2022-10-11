import numpy as np
import matplotlib.pyplot as plt
N = 10
L = 2*np.pi
dx = L/N

def v(x, i):
    xant = (i - 1)*dx
    xi = i*dx
    xsig = (i + 1)*dx
    if x > xant and x < xi:
        return (x - xant)/dx
    elif x > xi and x < xsig:
        return (xsig - x)/dx
    else:
        return 0

xx, dx = np.linspace(0, 2, N, retstep=True)
x = np.linspace(0, 2, 100)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(N):
    ax.plot(x, v(x, i))
    
