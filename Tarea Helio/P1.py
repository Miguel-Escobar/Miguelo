import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Definir log-pdf.

def logpdf(params, alpha):
    r1 = params[0]
    r2 = params[1]
    r12 = np.abs(r1 - r2)
    return 2*(-2*r1 - 2*r2 + .5*(r12/(1-alpha*r12)))

# Defino el paso metropolis: 

def paso_metropolis(params, alpha, retaccept=False):
    newparams = params + np.random.normal(loc=0.0, scale=1.0, size=len(params))
    logr = logpdf(newparams, alpha) - logpdf(params, alpha)
    accept = np.log(np.random.uniform(0,1))<logr
    if accept:
        if retaccept:
            return newparams
        else:
            return newparams, accept
    else:
        if retaccept:
            return params
        else:
            return params, accept

def METROPOLIS(ndatos, temper, initparams, alpha):
    resultados = np.zeros((ndatos, 2))
    params = initparams
    acceptance = 0
    for i in range(temper):
        params = paso_metropolis(params, alpha)
    for i in trange(ndatos):
        resultados[i] = params
        params, accepted = paso_metropolis(params, alpha, retaccept=True)
        acceptance += accepted
    return resultados
