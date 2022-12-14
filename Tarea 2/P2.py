import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma
x = np.array([-0.15, -0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1])

F = np.array([[np.sin(-0.15*2*np.pi), np.sin(-0.1*2*np.pi), np.sin(-0.05*2*np.pi), np.sin(0*2*np.pi), np.sin(0.1*2*np.pi), np.sin(0.2*2*np.pi), np.sin(0.3*2*np.pi), np.sin(0.4*2*np.pi), np.sin(0.6*2*np.pi), np.sin(0.7*2*np.pi), np.sin(0.8*2*np.pi), np.sin(0.9*2*np.pi), np.sin(1*2*np.pi)],
              [np.cos(-0.15*2*np.pi), np.cos(-0.1*2*np.pi), np.cos(-0.05*2*np.pi), np.cos(0*2*np.pi), np.cos(0.1*2*np.pi), np.cos(0.2*2*np.pi), np.cos(0.3*2*np.pi), np.cos(0.4*2*np.pi), np.cos(0.6*2*np.pi), np.cos(0.7*2*np.pi), np.cos(0.8*2*np.pi), np.cos(0.9*2*np.pi), np.cos(1*2*np.pi)]]).transpose()

y = np.array([180, 150, 80, 0, -110, -180, -182, -120, 120, 190, 198, 130, 0]).transpose()

sigmalista = np.array([5,6,7,2,3,10,11,10,5,6,3,7,11])

sigma = np.diag(sigmalista**2)

sinv = np.linalg.inv(sigma)

alpha = F.transpose() @ sinv @ F

alphainv = np.linalg.inv(F.transpose() @ sinv @ F)

b = (F.transpose() @ sinv @ y)

parametros = alphainv @ b 

A = parametros[0]
B = parametros[1]

def ajuste(t):
    return A*np.sin(2*np.pi*t) + B*np.cos(2*np.pi*x)

chi2 = np.sum(((y-ajuste(x))**2)/sigmalista**2)
chindof = chi2/11

p = gammainc(11/2, chi2/2)/gamma(11/2)

plt.figure(1)
plt.clf()
plt.plot(x, ajuste(x), label="Ajuste A = " + str(np.round(A, decimals=2)) + ', B = ' + str(np.round(B, decimals=2)))
plt.errorbar(x, y, sigmalista, fmt='o', ms=4, label='Datos')
plt.xlabel('Tiempo normalizado')
plt.ylabel('Velocidad Radial [km/s]')
plt.legend()
plt.show()
