#Datos de rayos gamma

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Función modelo:
E0 = 3

def modelo(E, N, alpha1, alpha2):
    cosa = N*E*E/((E/E0)**alpha1 + (E/E0)**alpha2)
    return cosa

# Energías
energia = np.array([0.2, 0.26, 0.32, 0.37, 0.43, 0.54, 0.68, 0.8, 1.0, 1.2, 1.4, 1.8,
                    2.1, 2.6, 3.1, 3.8, 4.6, 5.5, 6.9, 8.2, 10., 12., 14., 16.])
# Flujos
flujo_e2 = np.array([3.3, 3.15, 3.04, 2.7, 2.85, 2.65, 2.9, 2.75, 2.8, 2.65, 2.2,
                     2.0, 2.0, 1.95, 1.5, 1.4, 1.3, 1.0, 0.9, 1.1, 0.56, 0.54, 0.61, 0.33])

# Errores estándar
sigma = np.array([0.15, 0.12, 0.13, 0.14, 0.13, 0.13, 0.14, 0.13, 0.13, 0.15, 0.18,
                  0.2, 0.2, 0.22, 0.21, 0.19, 0.21, 0.18, 0.21, 0.2, 0.25, 0.3, 0.27, 0.16])

params, cov = curve_fit(modelo, energia, flujo_e2, sigma=sigma, method='lm')
# Gráfico en log-log
plt.figure()
plt.clf()
plt.errorbar(energia, flujo_e2, color='red', marker='o', linestyle='None', yerr=sigma)
plt.plot(np.linspace(0.2, 16, 100), modelo(np.linspace(0.2, 16, 100), params[0], params[1], params[2]))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$E$ (TeV)", fontsize=12)
plt.ylabel("flux ($10^{-12}$TeV cm$^{-2}$s$^{-1}$)", fontsize=12)
plt.savefig("Flujo_rayosgamma.pdf")
plt.show()
alpha = np.linalg.inv(cov)
alphamar = np.array([[alpha[0,0]-alpha[1,0]*alpha[1,0]/alpha[1,1], alpha[0,2] - alpha[1,0]*alpha[1,2]/alpha[1,1]], [alpha[2,0] - alpha[1,2]*alpha[1,0]/alpha[1,1], alpha[2,2] - alpha[1,2]*alpha[1,2]/alpha[1,1]]])

# Aproximación gaussiana me permite sacar las regiones de confianza con:

chi68 = -2 * np.log(1-.68)
chi95 = -2 * np.log(1-.95)

npoints = 50
x = np.linspace(params[0] - 4*np.sqrt(cov[0,0]), params[0] + 4*np.sqrt(cov[0, 0]), npoints)
y = np.linspace(params[2] - 4*np.sqrt(cov[2,2]), params[2] + 4*np.sqrt(cov[2, 2]), npoints)

X, Y = np.meshgrid(x, y)
Z = np.zeros((npoints, npoints))
Z2 = np.zeros((npoints, npoints))

deltax = X - params[0]
deltay = Y - params[2]

# Para sacar curvas de nivel:

def chi2(X1, Y1):
    return np.sum((flujo_e2 - modelo(energia, X1, params[1], Y1))**2/sigma**2)

# Relleno ambos Z:

for i in range(npoints):
    for j in range(npoints):
        Z[i, j] = deltax[i,j] * alphamar[0,0] * deltax[i,j] + deltay[i,j] * alphamar[1,0] * deltax[i,j] + deltax[i,j] * alphamar[0,1] * deltay[i,j] + deltay[i,j] * alphamar[1,1] * deltay[i,j]
        Z2[i, j] = chi2(X[i,j], Y[i,j])

# Ploteo

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(121)
ax.contourf(X, Y, Z, [0, chi68, chi95], cmap='Paired')
ax.plot(params[0], params[2], marker='o', color = 'r')
ax.set_xlabel(r"N [$10^{-12}\cdot TeV \cdot cm^{-2} \cdot s^{-1}$]")
ax.set_ylabel(r"$\alpha_{2}$")

ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z2)
ax2.plot(params[0], params[2], marker='o', color = 'r')
ax2.set_xlabel(r"N [$10^{-12}\cdot TeV \cdot cm^{-2} \cdot s^{-1}$]")
ax2.set_ylabel(r"$\alpha_{2}$")

fig.tight_layout()
fig.show()




