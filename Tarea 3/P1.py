import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma
xdata = np.array([-0.15, -0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1])

F = np.array([[np.sin(-0.15*2*np.pi), np.sin(-0.1*2*np.pi), np.sin(-0.05*2*np.pi), np.sin(0*2*np.pi), np.sin(0.1*2*np.pi), np.sin(0.2*2*np.pi), np.sin(0.3*2*np.pi), np.sin(0.4*2*np.pi), np.sin(0.6*2*np.pi), np.sin(0.7*2*np.pi), np.sin(0.8*2*np.pi), np.sin(0.9*2*np.pi), np.sin(1*2*np.pi)],
              [np.cos(-0.15*2*np.pi), np.cos(-0.1*2*np.pi), np.cos(-0.05*2*np.pi), np.cos(0*2*np.pi), np.cos(0.1*2*np.pi), np.cos(0.2*2*np.pi), np.cos(0.3*2*np.pi), np.cos(0.4*2*np.pi), np.cos(0.6*2*np.pi), np.cos(0.7*2*np.pi), np.cos(0.8*2*np.pi), np.cos(0.9*2*np.pi), np.cos(1*2*np.pi)]]).transpose()

ydata = np.array([180, 150, 80, 0, -110, -180, -182, -120, 120, 190, 198, 130, 0]).transpose()

sigmalista = np.array([5,6,7,2,3,10,11,10,5,6,3,7,11])

sigma = np.diag(sigmalista**2)

sinv = np.linalg.inv(sigma)

alpha = F.transpose() @ sinv @ F

alphainv = np.linalg.inv(F.transpose() @ sinv @ F)

b = (F.transpose() @ sinv @ ydata)

parametros = alphainv @ b 

# OKOKOKOK AHORA LO QUE HACEMOS ES::::::::

# Creación de Grid: 
npoints = 50
x = np.linspace(parametros[0] - 3*np.sqrt(alphainv[0,0]), parametros[0] + 3*np.sqrt(alphainv[0, 0]), npoints)
y = np.linspace(parametros[1] - 3*np.sqrt(alphainv[1,1]), parametros[1] + 3*np.sqrt(alphainv[1, 1]), npoints)

X, Y = np.meshgrid(x, y)
Z = np.zeros((npoints, npoints))

deltax = X - parametros[0]
deltay = Y - parametros[1]

for i in range(npoints):
    for j in range(npoints):
        Z[i, j] = deltax[i,j] * alpha[0,0] * deltax[i,j] + deltay[i,j] * alpha[1,0] * deltax[i,j] + deltax[i,j] * alpha[0,1] * deltay[i,j] + deltay[i,j] * alpha[1,1] * deltay[i,j]

chi68 = -2 * np.log(1-.68)
chi95 = -2 * np.log(1-.95)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(111)
#ax.contourf(X, Y, Z, [0,chi68])
ax.contourf(X, Y, Z, [0, chi68, chi95], cmap=None)
fig.show()