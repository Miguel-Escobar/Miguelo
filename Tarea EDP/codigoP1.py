import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.linalg import solve

N_temporales = 101
gamma = 1
puntos = 500
tau = 1/(4*np.pi**2*gamma)
tfinal = tau 
dt = tfinal/N_temporales

# defino cosas:

x, dx = np.linspace(0, 1, puntos, retstep=True)
s = gamma*dt/(2*dx**2)


t_0 = 1 - np.cos(2*np.pi*x)
limitesgrafico = (-0.05, 2.05)  # Para el caso de la animación


# Creacion de matrices:

diagonal_LHS = np.ones(puntos)*(2*s + 1)
diagonal_RHS = np.ones(puntos)*(1-2*s) 
bandalateral_LHS = np.ones(puntos-1)*(-s)
bandalateral_RHS = np.ones(puntos-1)*s

S_izq = diags([bandalateral_LHS, diagonal_LHS, bandalateral_LHS],
              offsets=[1, 0, -1]).toarray()
S_der = diags([bandalateral_RHS, diagonal_RHS, bandalateral_RHS],
              offsets=[1, 0, -1]).toarray()

# Condiciones de borde:

S_izq[0,0] = 1 + s
S_izq[-1,-1] = 1 + s
S_der[0,0] = 1 - s
S_der[-1,-1] = 1 - s

# Voy a crear un array donde guardar las soluciones:

solucion = np.zeros((N_temporales, puntos))

# Setear condicion inicial y de borde:

solucion[0, :] = t_0

for i in range(1, N_temporales):
    b = S_der @ solucion[i-1]
    solucion[i] = solve(S_izq, b)

# Funcion para la animación:

def funcAnimate(i):
    line.set_ydata(solucion[i, :])
    time.set_text(time_template % (i*dt))
    return line, time


if input('¿Desea verlo animado? (si/no): ').lower() == 'si':
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x [U.D.]')
    ax.set_ylabel('n(x,t)')
    ax.set_ylim(limitesgrafico)
    line, = ax.plot(x, solucion[0, :])
    time = ax.text(0.05, 0.1, '', transform=ax.transAxes)
    time_template = 't = %.1f [U.T.]'
    animation = FuncAnimation(fig, funcAnimate, frames=N_temporales,
                              interval=6000/N_temporales, blit=True)
else:
    for i in range(5):
        fig = plt.figure(i)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [U.D.]')
        ax.set_ylabel('n(x,t)')
        index = int(N_temporales/4)*i
        ax.plot(x, solucion[index, :], label="Crank-Nicolson")
        ax.plot(x, 1-np.cos(2*np.pi*x)*np.exp(-dt*index/tau), ls="--", label="Teorica")
        ax.set_ylim(0, 2.1)
        ax.legend()
        fig.show()

plt.show()
