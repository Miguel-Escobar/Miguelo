#################################################
## Codigo Particle Mesh: caso iones congelados ##
#################################################
import numpy as np
import matplotlib.pyplot as plt
import math
from random import seed
from random import random
from pylab import imshow, colorbar, title, show

N=120
#M=3*N
#w=2./(1+math.pi/N)
Nepc=10   # número de electrones por celda
v1 = np.empty([N,N])
ex = np.empty([N,N])
ey = np.empty([N,N])
rho = np.empty([N,N])
rx = np.empty(N*N*Nepc)
ry = np.empty(N*N*Nepc)
exp = np.empty(N*N*Nepc)
eyp = np.empty(N*N*Nepc)
vx = np.empty(N*N*Nepc)
vy = np.empty(N*N*Nepc)
omegape = 0.1
Pasos = 20
UEx = np.empty(Pasos)
UEy = np.empty(Pasos)
UKx = np.empty(Pasos)
UKy = np.empty(Pasos)
tiempo = np.empty(Pasos)

fft_v = np.empty([N,N], dtype=complex)
k2_arr = np.empty([N,N])

######################################
### INICIALIZACIÓN DE POSICIONES Y ###
### VELOCIDADES DE LAS PARTÍCULAS. ###
######################################
### (No es necesario inicializar los campos, pues en electrostática
### estos están dados por la densidad de partículas en cada instante). 

### Caso más sencillo: electrones "frios" ###
#vx.fill(0.)
vy.fill(0.)

### Electrones repartidos homogeneamente en la caja ###  
seed(1)
for i in range(N*N*Nepc):
    rx[i] = N*random()
seed(2)
for i in range(N*N*Nepc):
    ry[i] = N*random()

seed(3)
for i in range(N*N*Nepc):
    dado = random()
    if dado < .5:
        vx[i] = .4
    else:
        vx[i] = -.4
temp = .5*vx[0]**2



### Acá empezamos las iteraciones ###
for p in range(Pasos):
#    print('p=',p)
### Acá reingresamos las partículas que se nos pueden haber caido de la caja 
### (suponiendo condiciones de borde periódicas)
    for k in range(N*N*Nepc):
        if rx[k] >= N:
            rx[k] = rx[k] - N
        if ry[k] >= N:
            ry[k] = ry[k] - N
        if rx[k] < 0:
            rx[k] = rx[k] + N
        if ry[k] < 0:
            ry[k] = ry[k] + N
##########################
### DEPOSITO DE CARGAS ###
##########################
### Acá depositamos las cargas. 
### rho1 es un arreglo auxiliar de densidades, con 1 fila y
### 1 columna más que el verdadero arreglo de densidades rho.
# El uso de %N permite usar Condiciones de borde periódicas de manera compacta
    rho.fill(0.)
    for k in range(N*N*Nepc):
        i = int(rx[k])
        j = int(ry[k])
        dx = 1. + i -rx[k]
        dy = 1. + j -ry[k]
        rho[i,j]     = rho[i,j]     + dx*dy
        rho[(i+1)%N,j]   = rho[(i+1)%N,j]   + (1.-dx)*dy
        rho[i,(j+1)%N]   = rho[i,(j+1)%N]   + dx*(1.-dy)
        rho[(i+1)%N,(j+1)%N] = rho[(i+1)%N,(j+1)%N] + (1.-dx)*(1.-dy)
## Acá calculamos la densidad total considerando la fórmula vista
### en clases y que los electrones tienen carha negativa
    for i in range(N):
        for j in range(N):
            rho[i,j] = 1. - rho[i,j]/Nepc 

# Acá graficamos las densidades encontradas para cada tiempo   
    plt.figure(figsize=(10,10))
    im = imshow(rho)
    plt.colorbar(im)
    if p < 10:
        plt.savefig('rho_ts00'+str(p), dpi=None)
    else:
        if p < 100:
            plt.savefig('rho_ts0'+str(p), dpi=None)
        else:
            plt.savefig('rho_ts'+str(p), dpi=None)
    plt.close()    
#####################################################    
###        RESOLVEMOS ECUACIÓN DE POISSON         ###
### (En este ejemplo usamos método de relajación) ###
#####################################################

    fft_rho = np.fft.fft2(rho)
    num_onda0 = np.fft.fftfreq(fft_rho.shape[0],d=1)
    num_onda1 = np.fft.fftfreq(fft_rho.shape[1],d=1)

    for i in range(N):
        for j in range(N):
            k2_arr[i,j] = 4.*(np.sin(np.pi*num_onda0[i])**2 + np.sin(np.pi*num_onda1[j])**2)
        
    for i in range(N):
        for j in range(N):
            if k2_arr[i,j] > 0:
                fft_v[i,j] = fft_rho[i,j]/k2_arr[i,j]
            else:
                fft_v[i,j] = fft_rho[i,j]        
    v1 = np.real(np.fft.ifft2(fft_v))

### Acá calculamos los campos eléctricos centrado en el espacio
    for i in range(N):
        for j in range(N):
            ex[i,j] = -.5*((v1[(i+1)%N,j]+v1[(i+1)%N,(j+1)%N])-(v1[i,j]+v1[i,(j+1)%N]))
            ey[i,j] = -.5*((v1[i,(j+1)%N]+v1[(i+1)%N,(j+1)%N])-(v1[i,j]+v1[(i+1)%N,j]))

### Acá realizamos la interpolación
# El uso de %N permite usar Condiciones de borde periódicas de manera compacta
# De la forma en que i y j se calculan, los resultados pueden dar N, por eso es necesario hacer i%N y j%N, 
# para que en esos casos se mueva a cero por la CBP
    for k in range(N*N*Nepc):
        i = int(rx[k] + .5)
        j = int(ry[k] + .5)
        dx = 1. + i -rx[k] - .5
        dy = 1. + j -ry[k] - .5
        exp[k] = ex[(i-1)%N,(j-1)%N]*dx*dy + ex[i%N,(j-1)%N]*(1.-dx)*dy + ex[(i-1)%N,j%N]*dx*(1.-dy) + ex[i%N,j%N]*(1.-dx)*(1.-dy)
        eyp[k] = ey[(i-1)%N,(j-1)%N]*dx*dy + ey[i%N,(j-1)%N]*(1.-dx)*dy + ey[(i-1)%N,j%N]*dx*(1.-dy) + ey[i%N,j%N]*(1.-dx)*(1.-dy)

# Y avanzamos las velocidades y posiciones usando Verlet
        vx[k] = vx[k] - exp[k]*omegape**2
        vy[k] = vy[k] - eyp[k]*omegape**2
        rx[k] = rx[k] + vx[k]
        ry[k] = ry[k] + vy[k]  

# Finalmente acá calculamos las energías de los campos en los ejes x e y, 
# y las energías cinéticas de las partículas en los ejes x e y. 
    UKx[p] = np.sum(vx**2)/2.
    UKy[p] = np.sum(vy**2)/2.
    UEx[p] = (Nepc*omegape**2)*np.sum(ex**2)/2.
    UEy[p] = (Nepc*omegape**2)*np.sum(ey**2)/2.
    tiempo[p] = p*omegape/(2.*np.pi)
    
    print('p=',p,'UK',UKx[p],UKy[p],'UE',UEx[p],UEy[p])

    plt.figure(figsize=(10,10))
    
    nbins = 100
    E_min = np.min((vx**2+vy**2)/2.)
    E_max = np.max((vx**2+vy**2)/2.)
    E_bins = np.linspace(E_min, E_max, nbins)
    E_bins_center = np.empty(nbins-1)
    hist_E, bin_edges_E = np.histogram((vx**2+vy**2)/2., bins = E_bins)
    for i in range(nbins-1):
        E_bins_center[i] = .5*(E_bins[i]+E_bins[i+1])

    delta_E=E_bins[1]-E_bins[0]

    normalization_E = sum(hist_E)*delta_E
    hist_E = hist_E/normalization_E

    plt.plot(E_bins_center,hist_E,color='green', label='simulacion')
    plt.plot(E_bins_center,hist_E[0]*np.exp(-E_bins_center/temp)/np.exp(-E_bins_center[0]/temp),color='red', label='Maxwell-Boltzmann')
    plt.legend(loc='upper left', frameon=False, fontsize=18)
    plt.xlabel('E', size=18)
    plt.ylabel('dN/dE', size=18)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(0.001,2.5)
    plt.ylim(0.001,40)
    
    if p < 10:
        plt.savefig('spectrum00'+str(p), dpi=None)
    else:
        if p < 100:
            plt.savefig('spectrum0'+str(p), dpi=None)
        else:
            plt.savefig('spectrum'+str(p), dpi=None)
    plt.close()