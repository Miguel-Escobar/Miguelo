import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.integrate import nquad

# Definir log-pdf y pdf (sin normalizar).

def logpdf(Ra, alpha):
    r1 = np.linalg.norm(Ra[0])
    r2 = np.linalg.norm(Ra[1])
    r12 = np.linalg.norm(r1 - r2)
    return 2*(-2*r1 - 2*r2 + .5*(r12/(1+alpha*r12)))

def pdf(R, alpha):
    return np.exp(logpdf(R, alpha))

# Defino funciones para normalizar la pdf, de acuerdo con lo necesario para usar nquad:

def integral(alpha):
    #value, _ = nquad(integrable, [[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit]], args=[alpha], opts={"epsabs":10e1, "epsrel":10e1}) # Relajé mucho la tolerancia y aun así no corre
    value=1 # No estoy normalizando pq no me funciona.
    return value

def integrable(r1x,r1y,r1z,r2x,r2y,r2z, alpha):
    r1 = np.sqrt(r1x**2 + r1y**2 + r1z**2)
    r2 = np.sqrt(r1x**2 + r1y**2 + r1z**2)
    r12 = np.sqrt((r1x-r2x)**2+(r1y-r2y)**2+(r1z-r2z)**2)
    returnable = np.exp(-2*r1 - 2*r2 + .5*(r12/(1+alpha*r12)))**2
    # print("se llamó la funcion a integrar")
    return returnable

# Defino el paso metropolis: 

def paso_metropolis(R, alpha):    
    newR = R + np.random.normal(loc=0.0, scale=1.0, size=R.shape)
    logr = logpdf(newR, alpha) - logpdf(R, alpha)
    accept = np.log(np.random.uniform(0,1))<logr
    if accept:
        return newR, accept
    else:
        return R, accept

def METROPOLIS(ndatos, temper, initR, alpha, returnaccepted = False):
    resultados = np.zeros((ndatos,) + initR.shape)
    R = initR
    acceptance = 0
    for i in range(temper):
        R, basura = paso_metropolis(R, alpha)
    for i in range(ndatos):
        resultados[i] = R
        R, accepted = paso_metropolis(R, alpha)
        acceptance += accepted
    if returnaccepted:
        ratio = acceptance/ndatos
        return resultados, ratio
    else:
        return resultados

def energialocal(R, alpha):
    R1 = R[0]
    R2 = R[1]
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r12 = np.linalg.norm(r1 - r2)
    if r1 != 0 and r2 != 0 and r12 != 0:
        energy = -4 + np.dot(R1-R2, R1/r1-R2/r2)/(r12*(1+alpha*r12)**2) - 1/(r12*(1+alpha*r12)**3) - 1/(4*(1+alpha*r12)**4) + 1/r12
        return energy
    else:
        print("invalid value replaced by zero")
        return 0
        
def energiavariacional(alpha):
    sample, ratio = METROPOLIS(Ndatos, temper, R0, alpha, returnaccepted=True)
    #print("Metropolis corrió")
    normalizante = integral(alpha)
    #print("Normalización corrió")
    estimacion = np.zeros(Ndatos)
    for i in range(Ndatos):
        estimacion[i] = pdf(sample[i], alpha)*energialocal(sample[i], alpha)/normalizante
    return np.mean(estimacion), ratio

# AHORA TESTEO PARA VER SI CALCULA ENERGIAS VARIACIONALES:

Ndatos = 100000
temper = 1000
R0 = np.zeros((2, 3))+1 # Inicial al azar
limit = np.inf # La integral se lleva a cabo en (-limit,limit)^6 (si es np.inf es lo mismo que integrar en R^6)

test = energiavariacional(0.5)
print(test)

# Ahora lo corro para muchos alphas:
Nalphas = 40
alphas = np.linspace(0.01, 1, Nalphas)
EV = np.zeros(Nalphas)
Ratios = np.zeros(Nalphas)
for i in trange(len(alphas)):
    ev, ratio = energiavariacional(alphas[i])
    EV[i] = ev
    Ratios[i] = ratio

plt.plot(EV)
plt.plot(Ratios)

    

