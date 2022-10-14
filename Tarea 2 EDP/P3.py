import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from tqdm import trange
from scipy.sparse import diags
N = 100
L = 1
dx = L/(N)

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

