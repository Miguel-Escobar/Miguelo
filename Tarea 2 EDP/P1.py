import numpy as np
from scipy.linalg import solve_banded
N = 300
L = 1
R = L/4
V0 = 1
D = 100
dx = L/N

# Array con distancias desde el centro:

distancia_al_centro = np.zeros((N, N))
medio = int(N/2)
for i in range(N):
    for j in range(N):
        distancia_al_centro[i, j] = np.sqrt((medio - i)**2 + (medio - j)**2)*dx 


# Setear condiciones de borde:

def bordes(array, radius):
    copy = np.copy(array)
    copy[distancia_al_centro > radius] = -V0
    copy[0,:] = V0
    copy[:,0] = V0
    copy[-1,:] = V0
    copy[:,-1] = V0
    

    


