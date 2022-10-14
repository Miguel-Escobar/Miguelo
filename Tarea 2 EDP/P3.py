import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from tqdm import trange
from scipy.sparse import diags
N = 50
L = 1
dx = L/(N+1)

def v(x, i):
    xant = (i - 1)*dx
    xi = i*dx
    xsig = (i + 1)*dx
    returnable = np.zeros(len(x))
    returnable[np.logical_and(x > xant,x < xi)] = (x[np.logical_and(x > xant, x < xi)] - xant)/dx
    returnable[np.logical_and(x >= xi, x < xsig)] = (xsig - x[np.logical_and(x >= xi, x < xsig)])/dx
    return returnable

b = np.zeros(N)
x = np.zeros(N)

for i in trange(N):
    x[i] = (i+1)*dx
    xant = x[i] - dx
    xsig = x[i] + dx
    b[i] = (N/(4*np.pi**2)) * (2*np.sin(2*np.pi*x[i]) - np.sin(2*np.pi*xsig) - np.sin(2*np.pi*xant))

Adiag = np.ones(N)*(-2/dx)
Abandas = np.ones(N-1)*(1/dx)

A = diags([Abandas, Adiag, Abandas], offsets=[1, 0, -1]).toarray()

a = solve(A, b)

# Solución:

puntos = 200
xx, deltax = np.linspace(0, L, puntos, retstep=True)

un = np.zeros(puntos)

for i in range(N):
    un += a[i]*v(xx, i+1)

# Ahora FFT:
rho = np.sin(2*np.pi*xx)
rhofft = np.fft.fft(rho)
freq = np.fft.fftfreq(puntos, d=1)
LHS = np.zeros(puntos, dtype="complex_")
for i in range(puntos):
    if freq[i] == 0:
        LHS[i] = 0
    else:
        LHS[i] = (-rhofft[i]*deltax**2)/(4*(freq[i]**2)*(np.pi**2))

ufft = np.real(np.fft.ifft(LHS))



fig = plt.figure()
fig.clf()
ax = fig.add_subplot(111)
ax.plot(xx, un, label="FEM")
ax.plot(xx,ufft, label="FFT", ls="--")
# ax.plot(x, a, label="Vector A")
ax.legend()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"u(x)")
fig.show()
