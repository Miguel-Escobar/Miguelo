import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Función de integración por Leapfrog:

def leapfrog(rhs, initialvalue, initialtime, finaltime, Ndatapoints, params, forward=True):
    dt = (finaltime - initialtime)/(Ndatapoints)
    X = np.zeros((Ndatapoints, 2))
    if forward:
        X[0] = initialvalue
        t=0
        for i in range(Ndatapoints-1):
            X[i+1] = X[i] + rhs(X[i], t, params)*dt
            t += dt
    else:
        X[-1] = initialvalue
        t=finaltime
        for i in range(Ndatapoints-1):
            i = Ndatapoints-i-1
            X[i-1] = X[i] - rhs(X[i], t, params)*dt
            t -= dt
    return X

# RHS para este problema:

def atomohidrogeno(vec, t, l_E):
    l, E = l_E
    x1dot = vec[1]
    x2dot = vec[0]*(l*(l+1)/t - 2/t - E)
    return np.array([x1dot, x2dot])


# # Test:
# tinicial = 0
# tfinal = 10
# Ndatos = 1000
# x = leapfrog(atomohidrogeno,  np.array([1, 1]), tinicial, tfinal, Ndatos,np.array([0,-0.24277011108398425]), forward=False)[:, 0]
# plt.clf()
# plt.plot(x)
# plt.show()

