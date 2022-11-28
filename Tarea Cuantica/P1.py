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
    t, dt = np.linspace(initialtime, finaltime, Ndatapoints, retstep=True)
    X = np.zeros((Ndatapoints, 2))

    if forward:
        X[0] = initialvalue
        for i in range(Ndatapoints-1):
            X[i+1] = X[i] + rhs(X[i], t[i], params)*dt

    else:
        X[-1] = initialvalue
        for i in range(Ndatapoints-1):
            i = Ndatapoints-i-1
            X[i-1] = X[i] - rhs(X[i], t[i], params)*dt
    if returnTimes:
        return X, t
    else:
        return X

def numerov(R, S, x0, x1, initialtime, finaltime, Ndatapoints, params, rettimes=False):
    '''
    Integrador con metodo de numerov la ecuación x(t)'' - R(t)*x(t) = S(t).
    Necesita 2 condiciones dadas por la inicial y la inmediatamente
    siguiente. Retorna el array de la solución aproximada. R y
    S son funciones de t y parametros.
    '''
    x = np.zeros(Ndatapoints)
    x[0] = x0
    x[1] = x1
    t, dt = np.linspace(initialtime, finaltime, Ndatapoints, retstep=True)
    Sn = S(t[1], params)
    Slast = S(t[0], params)
    Rn = R(t[1], params)
    Rlast = R(t[0], params)
    for i in range(Ndatapoints-2):
        i += 1
        Snew = S(t[i+1], params)
        Rnew = R(t[i+1], params)
        coef1 = (1 - ((dt**2)/12)*Rnew)
        coef2 = 2*(1 + (5*(dt**2)/12)*Rn)
        coef3 = (1 - ((dt**2)/12)*Rlast)
        coefS = ((dt**2)/12)*(Snew + 10*Sn + Slast)
        x[i+1] = (coef2*x[i] - coef3*x[i-1] + coefS)/coef1
        Slast = Sn
        Sn = Snew
        Rlast = Rn
        Rn = Rnew
    if rettimes:
        return x, t
    else:
        return x


# Normalizar:

def normalize(x, fx):
    return fx/np.sqrt(simpson(fx**2, x))

#### AHORA EL CODIGO ES ESPECIFICO PARA ESTE PROBLEMA #####

tfinal = 30
tinicial = 0
Npasos = 1000
tol = 1e-4
l = 0
Emin = -1.1
Emax = -0.1
Nenergias = 100
# Condiciones iniciales convenientes:

def condicionesiniciales(E0, x0=0.01): # Esto antes era una función de E por eso es raro.
    return np.array([1, 0.01])

# RHS para este problema:

def atomohidrogeno(vec, t, E):
    x1dot = vec[1]
    x2dot = vec[0]*(l*(l+1)/(t**2) - 2/t - E)
    return np.array([x1dot, x2dot])


# Determinamos cuando cambia de signo:

def axisintersect(E, array=True):
    if array:  # Necesario pues la funcion len(E) pierde sentido si E es un float.
        intersect = np.zeros(len(E))
        for j in trange(len(E)):
            E0 = E[j]
            initialvalues = condicionesiniciales(E0)
            intersect[j] = leapfrog(atomohidrogeno, initialvalues, tinicial, tfinal, Npasos, E0, forward=False)[0, 0]
    else:
        initialvalues = condicionesiniciales(E)
        intersect = leapfrog(atomohidrogeno, initialvalues, tinicial, tfinal, Npasos, E, forward=False)[0, 0]
    return intersect

# Aplicación de las funciones:

energias = np.linspace(Emin, Emax, Nenergias) 
phiphi = axisintersect(energias)
fig1 = plt.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(111)

roots = []

for i in range(len(energias)-1):
    i+=1
    if phiphi[i]*phiphi[i-1] < 0: # Encontramos una autoenergía.
        root = bisect(axisintersect, energias[i-1], energias[i], args=False, xtol=tol)
        roots.append(root)
        x = np.linspace(tinicial, tfinal, Npasos)
        fx = leapfrog(atomohidrogeno, condicionesiniciales(root), tinicial, tfinal, Npasos, root, forward=False)[:, 0]
        ax1.plot(x, normalize(x, fx), label=("Energia = %.2f" % root) + "$\cdot 13.6$ eV" )

ax1.legend()
ax1.set_xlabel("x [$a_0$]")
ax1.set_ylabel("U(x)")
fig1.show()
print(roots)