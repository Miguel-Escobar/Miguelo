import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import solve

gamma = 1
puntos = 300
tau = 1/(4*np.pi**2*gamma)

x, dx = np.linspace(0, 1, puntos, retstep=True)
t_0 = 1 - np.cos(2*np.pi*x)
limitesgrafico = (-0.05, 2.05)

v0f = np.fft.fft(t_0)
freq = np.fft.fftfreq(puntos, d=1)
v0f_r = np.real(v0f)
v0f_i = np.imag(v0f)

def derivada(t, u):
    return gamma*np.fft.ifft(-4*np.pi**2*freq**2/(dx**2) * np.fft.fft(u))

obj = solve_ivp(derivada, (0, tau), t_0)

t = obj.t
y = obj.y

for i in range(4):
    fig = plt.figure(i)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x [U.D.]')
    ax.set_ylabel('n(x,t)')
    index = np.argmax(t > tau/4 * i)
    ax.plot(x, y[:, index], label="Pseudo-Espectral")
    ax.plot(x, 1-np.cos(2*np.pi*x)*np.exp(-t[index]/tau), ls="--", label="Teorica")
    ax.set_ylim(0, 2.1)
    ax.legend()
    fig.show()


fig = plt.figure(4)
fig.clf()
ax = fig.add_subplot(111)
ax.set_xlabel('x [U.D.]')
ax.set_ylabel('n(x,t)')
index = -1
ax.plot(x, y[:, index], label="Pseudo-Espectral")
ax.plot(x, 1-np.cos(2*np.pi*x)*np.exp(-t[index]/tau), ls="--", label="Teorica")
ax.set_ylim(0, 2.1)
ax.legend()
fig.show()





