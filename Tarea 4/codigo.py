import numpy as np
from scipy.special import gamma, loggamma
import matplotlib.pyplot as plt

# Definir Log-likelihood.

def ll(E, k, theta):
    primer_término = np.log(2*np.pi*np.sqrt(E)) + loggamma(k + 1)
    segundo_término = -(np.log(np.pi**(3/2) * theta**(3/2)) + (3/2)*np.log(k) + loggamma(k - 1/2))
    tercer_término = -(k + 1) * np.log(1 + E/(k*theta))
    return np.sum(primer_término + segundo_término + tercer_término)


# Importar datos:

data = np.genfromtxt("energias_electrones.csv")

# Defino el paso metropolis: 

def paso_metropolis(a, datos):
    acandidato=np.zeros(2)
    acandidato[0]=a[0] + np.random.normal(scale=1)
    acandidato[1]=a[1] + np.random.normal(scale=1)
    if acandidato[0] <= 0:
        acandidato[0] = a[0]
    if acandidato[1]<=0:
        acandidato[1] = a[1]

    logr = ll(datos,acandidato[0], acandidato[1]) - ll(datos,a[0], a[1])

    if np.log(np.random.uniform(0,1))<logr:
        return acandidato
    else:
        return a

def METROPOLIS(ndatos, datos, temper, a0):
    resultados = []
    params = a0
    for i in range(temper):
        params = paso_metropolis(params, datos)
    for i in range(ndatos):
        params = paso_metropolis(params, datos)
        resultados.append(params)
    return np.array(resultados)

def analiza(seriea):
    print("Promedios:",np.mean(seriea,axis=0))
    plt.figure()
    plt.clf()
    plt.hist2d(seriea[:,0],seriea[:,1], 100)
    plt.show()

result = METROPOLIS(100000, data, 2000, [3, 1])

analiza(result)
 