import numpy as np
import matplotlib.pyplot as plt
import math

# Parámetros y variables globales
# Unidades
sigma=1
masa=1
epsilon=1

# Parámetros globales
N=20 # Número de partículas
L=np.sqrt(N*sigma**2) # Tamaño de la caja
Temperatura=0.1 # Temperatura
Ttotal = 10 # Tiempo total de simulación
dt= 0.01 # Paso de tiempo
radioc=2.5*sigma # Radio de corte
radioc2=radioc*radioc # Radio de corte al cuadrado

# Variables globales
# Pongo x e y por separado, se podría también tener arreglos de dos columnas
x=np.zeros(N)
y=np.zeros(N)
vx=np.zeros(N)
vy=np.zeros(N)

# Funciones útiles

# Retorna la fuerza entre dos partículas, dada su distancia relativa
# Recibe dx,dy (ya corregidos por CBP)
# Retorna fx,fy
def fuerzapar(dx,dy):
    fx=XXXX
    fy=XXXX
    return fx,fy

# Calcula la distancia entre dos partículas y corrije por CBP
# Recibe los índices de las partículas
# Retorna dx,dy
def dij(i,j):
    dx=x[i]-x[j]
    dy=y[i]-y[j]
    if (dx>0.5*L):
        dx=dx-L
    if (dx<-0.5*L):
        dx=dx+L
    if (dy>0.5*L):
        dy=dy-L
    if (dy<-0.5*L):
        dy=dy+L        
    return dx,dy

# Termostato de reescalamiento de velocidades
# No recibe ningún parámetro ni retorna datos
# Modifica los arreglos vx,vy
def termostato():
    
# Inicializa las coordenadas y velocidades
# No recibe ningún parámetro ni retorna datos
# Modifica los arreglos x,y,vx,vy
def condicioninicial():
    

# Simulación completa

#Arreglo de aceleraciones
ax=np.zeros(N)
ay=np.zeros(N)
#Inicializa
condicioninicial()
#Loop de simulación
paso=0
for t in np.arange(0,Ttotal,dt):
    # Calcula las aceleraciones. Método O(N^2)
    for i in range(N):
        ax[i]=0
        ay[i]=0
        for j in range(N):
            dx,dy=dij(i,j)
            if (i!=j and dx*dx+dy*dy<radioc2):
                fx,fy=fuerzapar(dx,dy)
                ax[i]=ax[i]+fx
                ay[i]=ay[i]+fy
        #Se divide al final por la masa, para tener la aceleración
        ax[i]=ax[i]/masa
        ay[i]=ay[i]/masa
        
    # Integra con LeapFrog
    # Como son numpy arrays se puede hacer vectorizado también
    for i in range(N):
        vx[i]=vx[i]+dt*ax[i]
        vy[i]=vy[i]+dt*ay[i]
        x[i]=x[i]+dt*vx[i]
        y[i]=y[i]+dt*vy[i]
        
    # Corrige la CBP
    for i in range(N):
        if(x[i]>L):
            x[i]=x[i]-L
        if(x[i]<0):
            x[i]=x[i]+L
        if(y[i]>L):
            y[i]=y[i]-L
        if(y[i]<0):
            y[i]=y[i]+L 
    
    # Mide lo que haya que medir
    #####
    
    # Cada 1000 pasos, reescala la velocidades
    if(paso%1000==0):
        termostato()
        
    # Se incrementa en uno el paso
    paso=paso+1