import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
numbins = 100

result = np.genfromtxt("resultados.csv", delimiter=",")
k = result[:,0]
theta = result[:, 1]

def probability(height, array, area):
    return np.sum(array[array>=height] * area)

def heightfinder(probtarget, array, area):
    def aux(h, array, area):
        return probability(h, array, area)-probtarget
    root = root_scalar(aux, args=(array, area), x0=0, x1=10)
    root = root.root
    return root







fig = plt.figure("Histograma 2D")
fig.clf()
ax = fig.add_subplot(111)

alturas, kedges, thetaedges, _ = ax.hist2d(k, theta, bins=numbins, density=True)
indk, indtheta = np.unravel_index(np.argmax(alturas), alturas.shape)
deltak = kedges[1]-kedges[0]
deltatheta = thetaedges[1]-thetaedges[0]
area =deltak*deltatheta
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$\theta$")
ax.plot(kedges[indk]+deltak/2, thetaedges[indtheta]+deltatheta/2, 'o', label='Moda', color='r')
ax.legend()
fig.show()

# Region de confianza 95%:
topval = 10
h95 = heightfinder(.95, alturas, area)
alturas2=np.ma.masked_array(alturas, alturas < h95)
alturas2[alturas2 >= h95] = topval
X, Y = np.meshgrid(kedges, thetaedges)

fig2 = plt.figure("Region de confianza")
fig2.clf()
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(X, Y, alturas2.T)
fig2.show()