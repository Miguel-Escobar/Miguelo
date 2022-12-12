import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

# Parámetros y variables globales
# Unidades
sigma=1
masa=1
epsilon=1

# Parámetros globales
sqrtN=30
N = sqrtN*sqrtN
L=np.sqrt(N*sigma**2) # Tamaño de la caja
Temperatura=1 # Temperatura
Ttotal = 30 # Tiempo total de simulación
Transiente = 10
dt= 0.01 # Paso de tiempo
radioc=2.5*sigma # Radio de corte
radioc2=radioc*radioc # Radio de corte al cuadrado


# Funciones útiles

# Retorna la fuerza entre dos partículas, dada su distancia relativa
# Recibe dx,dy (ya corregidos por CBP)
# Retorna fx,fy
def fuerzapar(dx,dy):

    r = dx**2 + dy**2
    fx = -48*epsilon*((sigma**6)/(r**4) - (2*sigma**12)/(r**7))*dx
    fy = -48*epsilon*((sigma**6)/(r**4) - (2*sigma**12)/(r**7))*dy

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
    global vx, vy
    Tcinetica = np.mean((masa/2)*(vx**2 + vy**2))
    ajuste = np.sqrt(Temperatura/Tcinetica)
    vx = vx*ajuste
    vy = vy*ajuste
    return

# Inicializa las coordenadas y velocidades
# No recibe ningún parámetro ni retorna datos
# Modifica los arreglos x,y,vx,vy

def condicioninicial():
    global x, y, vx, vy
    vx = np.sqrt(2*Temperatura/masa)*norm.rvs(size=N)
    vy = np.sqrt(2*Temperatura/masa)*norm.rvs(size=N)
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    lin, espaciado = np.linspace(0, L, sqrtN, endpoint=False, retstep=True)
    x = np.meshgrid(lin,lin)[0].flatten() + espaciado/2
    y = np.meshgrid(lin,lin)[1].flatten() + espaciado/2
    return

# Curve fit:

def fittable(x, b):
    return b*x



# Variables globales
# Pongo x e y por separado, se podría también tener arreglos de dos columnas
x=np.zeros(N)
y=np.zeros(N)

vx=np.zeros(N)
vy=np.zeros(N)

#Arreglo de aceleraciones}

ax=np.zeros(N)
ay=np.zeros(N)

#Inicializa

condicioninicial()
termostato()

#Loop de simulación
paso=0
Npasos = int(Ttotal/dt)
Ntransiente = int(Transiente/dt)
listaanimable = []#[np.array([x.copy(), y.copy()])]
listavelocidades = [] #[np.array([vx.copy(), vy.copy()])]
for t in trange(Npasos + Ntransiente):
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
    if paso%100==0:
        termostato()
    # Se incrementa en uno el paso
    paso=paso+1

    if paso > Ntransiente:
        listaanimable.append(np.array([x.copy(), y.copy()]))
        listavelocidades.append(np.array([vx.copy(), vy.copy()]))

listaanimable = np.array(listaanimable)
listavelocidades = np.array(listavelocidades)

# Calculo del desplazamiento cuadratico medio:

desplazamientosxy = np.cumsum(listavelocidades*dt, axis=0)
rcuadraticomedio = np.mean(desplazamientosxy[:, 0, :]**2 + desplazamientosxy[:, 1, :]**2, axis=1)
tiempos = dt*np.arange(Npasos)
coeficienteD = rcuadraticomedio[1:]/(4*tiempos[1:])


adivinanza =  (rcuadraticomedio[-1]-rcuadraticomedio[0])/(tiempos[-1]-tiempos[0])
b, _ = curve_fit(fittable, tiempos, rcuadraticomedio, p0=adivinanza)


fig2 = plt.figure(2)
fig2.clf()
ax2 = fig2.add_subplot(211)
ax3 = fig2.add_subplot(212)
ax2.plot(tiempos, rcuadraticomedio, label=r"$\langle r(t)^{2} \rangle$")
ax2.plot(tiempos, fittable(tiempos, b), ls="--", label="Ajuste lineal")
ax3.plot(tiempos[1:], coeficienteD)
ax3.set_xlabel("tiempo")
ax2.set_xlabel("tiempo")
ax3.set_ylabel("D(t)")
ax2.set_ylabel(r"$\langle r(t)^{2} \rangle$")
ax2.legend()
fig2.tight_layout()
fig2.show()
print("Temperatura: " + str(Temperatura))
print("Coeficiente D: " +  str(coeficienteD[-1]))
print("Coeficiente ajuste lineal: " + str(b[0]))

# listaD = [1.622675119509858e-05, 0.0019038689202817295, 1.086578238614398e-05, 0.0005486093899001387, 0.00043602113583176514, 0.0008867940939022173, 0.002474892632365285, 0.0014321142393194145, 0.0007341196596340576, 0.003499715373845753, 0.0015031630698350061, 0.0013179581869484176]
# temperaturas = [.1, .1, .1, .5, .5, .5, 1, 1, 1, 2, 2, 2]
# fig = plt.figure(1)
# fig.clf()
# ax = fig.add_subplot(111)
# ax.plot(temperaturas, listaD, 'o', label="Datos")
# ax.plot([0.1, 0.5, 1, 2], [6.436e-4, 6.238e-4, 1.547e-3, 2.106e-3], "-o", label="Promedios")
# ax.legend()
# ax.set_xlabel("Temperatura")
# ax.set_ylabel("Coeficiente de difusión")
# fig.tight_layout()
# fig.show()

# while True:
#     if input("¿Guardaste el gráfico? ").lower() == "si":
#         break
#     else: 
#         print("Acuerdate de guardar el gráfico!")
