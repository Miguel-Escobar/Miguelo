import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import bisect
from scipy.integrate import simpson


def leapfrog(rhs, initialvalue, initialtime, finaltime, Ndatapoints, params, forward=True, returnTimes=False):
    '''
    Sea X: R -> R^n. Esta función resulve sistemas del estilo dX/dt = F(t, X) usando
    el método de Leapfrog (no staggered). Se puede integrar de adentro hacia afuera o
    de afuera hacia adentro con el parámetro forward.
    '''
    dt = (finaltime - initialtime)/(Ndatapoints-1)
    X = np.zeros((Ndatapoints, 2))

    if forward:
        X[0] = initialvalue
        t=initialtime
        times = [t]
        for i in range(Ndatapoints-1):
            X[i+1] = X[i] + rhs(X[i], t, params)*dt
            t += dt
            times.append(t)
    else:
        X[-1] = initialvalue
        t=finaltime
        for i in range(Ndatapoints-1):
            i = Ndatapoints-i-1
            X[i-1] = X[i] - rhs(X[i], t, params)*dt
            t -= dt
    return X

# Normalizar (OJO QUE ESTE NO ES EL QUE HAY QUE USAR PARA QUE LA FUNCION DE ONDA SEA MODULO CUADRADO INTEGRABLE):

def normalize(x, fx):
    return fx/simpson(fx, x)

#### AHORA EL CODIGO ES ESPECIFICO PARA ESTE PROBLEMA #####

tfinal = 30
Npasos = 1000
dt = tfinal/(Npasos-1)
l = 0

# RHS para este problema:

def atomohidrogeno(vec, t, E):
    x1dot = vec[1]
    x2dot = vec[0]*(l*(l+1)/(t**2) - 2/t - E)
    return np.array([x1dot, x2dot])

# Condiciones iniciales convenientes:

def condicionesiniciales(E0, x0=0.01):
    # k2_0 = (l*(l+1)/(tfinal**2) - 2/tfinal - E0)   # PODRIA USAR ESTO PERO ES COMO MAGIA NEGRA
    # v0 = -np.sqrt(k2_0)*x0 - k2_0*x0*dt/2
    # return np.array([x0, v0])
    return np.array([1, 0.01])
# Determinamos cuando cambia de signo:

def axisintersect(E, array=True):
    if array:  # Necesario pues la funcion len(E) pierde sentido si E es un float.
        intersect = np.zeros(len(E))
        for j in trange(len(E)):
            E0 = E[j]
            initialvalues = condicionesiniciales(E0)
            intersect[j] = leapfrog(atomohidrogeno, initialvalues, 0, tfinal, Npasos, E0, forward=False)[0, 0]
    else:
        initialvalues = condicionesiniciales(E)
        intersect = leapfrog(atomohidrogeno, initialvalues, 0, tfinal, Npasos, E, forward=False)[0, 0]
    return intersect

# Aplicación de las funciones:

energias = np.linspace(-1.01, -2/tfinal, 100) # Se llega hasta -2/tfinal pues más allá la función de condiciones iniciales se indefine.
phiphi = axisintersect(energias)
fig1 = plt.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(111)

roots = []

for i in range(len(energias)-1):
    i+=1
    if phiphi[i]*phiphi[i-1] < 0: # Encontramos una autoenergía.
        root = bisect(axisintersect, energias[i-1], energias[i], args=False, xtol=1e-4)
        roots.append(root)
        x = np.linspace(0, tfinal, Npasos)
        fx = leapfrog(atomohidrogeno, condicionesiniciales(root), 0, tfinal, Npasos, root, forward=False)[:, 0]
        ax1.plot(x, normalize(x, fx), label=("Energia = %.2f" % root) + "$\cdot 13.6$ eV" )

ax1.legend()
ax1.set_xlabel("x [$a_0$]")
ax1.set_ylabel("U(x)")
fig1.show()
print(roots)