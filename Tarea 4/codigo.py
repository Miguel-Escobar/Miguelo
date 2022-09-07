from re import S
import numpy as np
from scipy.special import gamma

# Definir Log-likelihood.

def ll(E, k, theta):
    primer_término = np.log(2*np.pi*np.sqrt(E)*gamma(k + 1))
    segundo_término = -np.log(np.pi**(3/2) * theta**(3/2) * gamma(k - 1/2) * k**(1/3))
    tercer_término = -(k + 1) * np.log(1 + E/(k*theta))
    return primer_término + segundo_término + tercer_término

# Importar datos:

data = np.genfromtxt("energias_electrones.csv")

# 