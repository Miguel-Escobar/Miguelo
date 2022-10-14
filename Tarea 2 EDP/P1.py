import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

N = int(input("Grilla de N x N, N="))
L = 10
r = L/4
V0 = 10
D = 100
dx = L/N
w = 2/(1 + np.pi/N)
miniter = 10
maxiter = 1000 - miniter

# Array con distancias desde el centro:

distancia_al_centro = np.zeros((N, N))
medio = int(N/2)-1
for i in range(N):
    for j in range(N):
        distancia_al_centro[i, j] = np.sqrt((medio - i + 1/2)**2 + (medio - j + 1/2)**2)*dx 

# Setear condiciones de borde:

def bordes(array):
    array[distancia_al_centro < r] = -V0
    array[0,:] = V0
    array[:,0] = V0
    array[-1,:] = V0
    array[:,-1] = V0
    return

# test región interior:

def region_interior(i, j):
    return i < N-1 and i > 0 and j < N-1 and j > 0 and distancia_al_centro[i, j] > r

# iteración Gauss-Seidel:

def gs(array):
    length, width = array.shape
    for i in range(length):
        for j in range(width):
            if region_interior(i, j):
                array[i, j] = (array[i, j - 1] + array[i, j + 1] + array[i - 1, j] + array[i + 1, j])/4
    return

# sobrerelajación:

def sobrerelax(array):
    length, width = array.shape
    for i in range(length):
        for j in range(width):
            if region_interior(i, j):
                array[i, j] = (1-w)*array[i, j] + (array[i, j - 1] + array[i, j + 1] + array[i - 1, j] + array[i + 1, j])*w/4 
    return

# test de convergencia (cortesía de prof. Valentino en curso de met num para la ciencia):

def convergio(phi, phi_anterior, rtol=0.1):
    not_zero = phi_anterior != 0
    dif_relativa = ((phi_anterior[not_zero] - phi[not_zero]) / phi_anterior[not_zero])
    return np.fabs(dif_relativa).max() < rtol

# Para calcular la capacitancia:

def capacitancia(phi, V0, D):
    E = -np.gradient(phi, dx)[0]
    Q = simps(E[0], dx=dx)*D/np.pi
    Cap = Q/(2*V0)
    return Cap

# Resolución:

sol = np.zeros((N,N))
bordes(sol) # Seteo condiciones de borde
counter = 0

for i in range(miniter): # Mínimo de iteraciones
    sobrerelax(sol)

while True: 
    solcopy = sol.copy() # Para comparar la convergencia
    sobrerelax(sol)
    counter += 1
    if (counter > 1 and convergio(sol, solcopy)) or counter > maxiter:
        break

fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(sol)
ax.set_xlabel("SOR")
fig.show()

print("Capacitancia = " + str(capacitancia(sol, V0, D)))